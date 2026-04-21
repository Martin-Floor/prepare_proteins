from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import html
import http.cookiejar
import json
import mimetypes
import os
import re
import shutil
import tarfile
import time
import uuid
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from .base import ParameterizationBackend, ParameterizationResult, register_backend

_build_opener = urllib_request.build_opener
_urlopen = urllib_request.urlopen

_QUICK_BILAYER_OPTION_KEYS = {
    "upper",
    "lower",
    "membtype",
    "margin",
    "wdist",
    "ion_conc",
    "ion_type",
    "prot_projection_upper",
    "prot_projection_lower",
    "ppm",
    "run_ppm",
    "topologyIn",
    "heteroatoms",
    "clone_job",
    "run_ffconverter",
}
_MEMBRANE_BUILDER_OPTION_KEYS = {
    "upper",
    "lower",
    "lateral_length",
    "hetero_lx",
    "wdist",
    "hetero_wdist",
    "hetero_xy_option",
    "hetero_z_option",
    "check_penetration",
    "source_jobid",
    "source_project",
}
_PDB_READER_OPTION_KEYS = {
    "jobid",
    "pdb_id",
    "source",
    "upload_path",
    "pdb_format",
    "correct_pdb",
    "include_hetero",
    "include_water",
    "include_dna",
    "include_rna",
    "system_pH",
    "preserve_hydrogens",
    "mutations",
}

_DONE_STATUSES = {"done", "finished", "complete", "completed"}
_RUNNING_STATUSES = {"pending", "running", "submitted", "queued", "compressing"}
_RUNNING_STATUS_TOKENS = tuple(sorted(_RUNNING_STATUSES))
_ERROR_STATUS_TOKENS = ("error", "fail", "abort")
_WORKFLOW_MODES = {"full", "submit_only", "collect"}
_TOPOLOGY_SUFFIXES = (".prmtop", ".parm7")
_COORDINATE_SUFFIXES = (".inpcrd", ".rst7")
_CHAIN_FIELD_PATTERN = re.compile(r"^chains\[([^\]]+)\]\[([^\]]+)\]$")
_RETRIEVER_LINK_PATTERN = re.compile(
    r"""\?doc=input/(?P<doc>[^&"'#]+)&step=(?P<step>\d+)&project=(?P<project>[^&"'#]+)&jobid=(?P<jobid>\d+)""",
    re.IGNORECASE,
)
_PROTEIN_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "CYM",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "HSD",
    "HSE",
    "HSP",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "ACE",
    "NME",
}
_PROTEIN_RESIDUE_EQUIVALENTS = {
    "HIS": "HIS",
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
}


class _HTMLFormParser(HTMLParser):
    def __init__(self, *, target_id: Optional[str] = None, target_name: Optional[str] = None) -> None:
        super().__init__()
        self.target_id = target_id
        self.target_name = target_name
        self.forms: List[Dict[str, Any]] = []
        self._current_form: Optional[Dict[str, Any]] = None
        self._current_select: Optional[Dict[str, Any]] = None
        self._current_option: Optional[Dict[str, Any]] = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attrs_dict = {key: value for key, value in attrs}
        if tag == "form":
            form = {
                "attrs": attrs_dict,
                "inputs": [],
                "selects": [],
            }
            self.forms.append(form)
            self._current_form = form
            return
        if self._current_form is None:
            return
        if tag == "input":
            self._current_form["inputs"].append(attrs_dict)
            return
        if tag == "select":
            select = {"attrs": attrs_dict, "options": []}
            self._current_form["selects"].append(select)
            self._current_select = select
            return
        if tag == "option" and self._current_select is not None:
            option = {"attrs": attrs_dict, "text": ""}
            self._current_select["options"].append(option)
            self._current_option = option

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self._current_form = None
            return
        if tag == "select":
            self._current_select = None
            return
        if tag == "option":
            self._current_option = None

    def handle_data(self, data: str) -> None:
        if self._current_option is not None:
            self._current_option["text"] += data

    def matching_form(self) -> Optional[Dict[str, Any]]:
        for form in self.forms:
            attrs = form["attrs"]
            if self.target_id and attrs.get("id") == self.target_id:
                return form
            if self.target_name and attrs.get("name") == self.target_name:
                return form
        return self.forms[0] if self.forms else None


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[Dict[str, Any]]] = []
        self._current_row: Optional[List[Dict[str, Any]]] = None
        self._current_cell: Optional[Dict[str, Any]] = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attrs_dict = {key: value for key, value in attrs}
        if tag == "tr":
            self._current_row = []
            return
        if tag in {"td", "th"} and self._current_row is not None:
            self._current_cell = {
                "attrs": attrs_dict,
                "text": "",
                "links": [],
            }
            return
        if tag == "a" and self._current_cell is not None:
            href = attrs_dict.get("href")
            if href:
                self._current_cell["links"].append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._current_row is not None and self._current_cell is not None:
            self._current_cell["text"] = self._current_cell["text"].strip()
            self._current_row.append(self._current_cell)
            self._current_cell = None
            return
        if tag == "tr" and self._current_row is not None:
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell["text"] += data


def _looks_like_mapping_of_models(option: Any, known_keys: Iterable[str]) -> bool:
    if not isinstance(option, Mapping):
        return False
    option_keys = {str(key).strip() for key in option.keys()}
    return not bool(option_keys & set(known_keys))


def _resolve_model_option(option: Any, model_name: str, known_keys: Iterable[str] = ()) -> Any:
    if _looks_like_mapping_of_models(option, known_keys):
        return option.get(model_name)
    return option


def _normalize_bool(value: Any, *, default: Optional[bool] = None) -> Optional[bool]:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Could not interpret boolean value {value!r}.")


def _build_mutation_entries(mutations: Any) -> list[Dict[str, str]]:
    """Validate a pdb_reader['mutations'] list and return normalised form-field triples.

    Each returned entry is a dict with keys 'chain' (CHARMM segid), 'rid' (resid as
    string), and 'patch' (3-letter target residue name). Input specs must be mappings
    with keys 'chain', 'resid', 'target'.
    """
    if mutations is None:
        return []
    if isinstance(mutations, (str, bytes, Mapping)) or not isinstance(mutations, Sequence):
        raise ValueError(
            "CHARMM-GUI pdb_reader['mutations'] must be a list of mapping entries."
        )
    entries: list[Dict[str, str]] = []
    for i, spec in enumerate(mutations):
        if not isinstance(spec, Mapping):
            raise ValueError(
                f"CHARMM-GUI pdb_reader['mutations'][{i}] must be a mapping with keys "
                "'chain', 'resid', 'target'."
            )
        missing = [k for k in ("chain", "resid", "target") if k not in spec]
        if missing:
            raise ValueError(
                f"CHARMM-GUI pdb_reader['mutations'][{i}] missing keys: {missing}."
            )
        chain = str(spec["chain"]).strip()
        if not chain:
            raise ValueError(
                f"CHARMM-GUI pdb_reader['mutations'][{i}]: 'chain' must be a non-empty string."
            )
        try:
            resid = int(spec["resid"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"CHARMM-GUI pdb_reader['mutations'][{i}]: 'resid' must be an integer."
            ) from exc
        target = str(spec["target"]).strip().upper()
        if not target:
            raise ValueError(
                f"CHARMM-GUI pdb_reader['mutations'][{i}]: 'target' must be a non-empty string."
            )
        if target not in _PROTEIN_RESIDUES:
            raise ValueError(
                f"CHARMM-GUI pdb_reader['mutations'][{i}]: 'target'={target!r} is not a "
                f"recognised CHARMM patch. Expected one of {sorted(_PROTEIN_RESIDUES)}."
            )
        entries.append({"chain": chain, "rid": str(resid), "patch": target})
    return entries


def _normalize_workflow_mode(value: Any) -> str:
    text = str(value or "full").strip().lower()
    if not text:
        text = "full"
    if text not in _WORKFLOW_MODES:
        raise ValueError(
            f"Unsupported CHARMM-GUI workflow_mode {value!r}. "
            f"Expected one of: {', '.join(sorted(_WORKFLOW_MODES))}."
        )
    return text


def _parse_lipid_ratio_string(value: Any) -> Dict[str, str]:
    text = str(value or "").strip()
    if not text:
        raise ValueError("CHARMM-GUI lipid composition strings must be non-empty.")

    if text.count("=") == 1:
        lipid_names_part, ratio_part = text.split("=", 1)
        lipid_names = [token.strip().lower() for token in lipid_names_part.split(":") if token.strip()]
        ratio_values = [token.strip() for token in ratio_part.split(":") if token.strip()]
        if lipid_names and ratio_values and len(lipid_names) == len(ratio_values):
            composition: Dict[str, str] = {}
            for lipid_name, amount_text in zip(lipid_names, ratio_values):
                try:
                    numeric_amount = float(amount_text)
                except ValueError as exc:
                    raise ValueError(
                        f"Could not parse CHARMM-GUI lipid composition amount {amount_text!r} in {text!r}."
                    ) from exc
                if numeric_amount < 0:
                    raise ValueError("CHARMM-GUI lipid composition amounts must be non-negative.")
                composition[lipid_name] = amount_text
            return composition

    composition: Dict[str, str] = {}
    for item in text.split(":"):
        token = item.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"Could not parse CHARMM-GUI lipid composition token {token!r}. "
                "Expected entries like 'POPC=7:POPE=3'."
            )
        lipid_name, amount = token.split("=", 1)
        lipid_key = lipid_name.strip().lower()
        amount_text = amount.strip()
        if not lipid_key or not amount_text:
            raise ValueError(
                f"Could not parse CHARMM-GUI lipid composition token {token!r}. "
                "Expected entries like 'POPC=7:POPE=3'."
            )
        try:
            numeric_amount = float(amount_text)
        except ValueError as exc:
            raise ValueError(
                f"Could not parse CHARMM-GUI lipid composition amount {amount_text!r} in token {token!r}."
            ) from exc
        if numeric_amount < 0:
            raise ValueError("CHARMM-GUI lipid composition amounts must be non-negative.")
        composition[lipid_key] = amount_text

    if not composition:
        raise ValueError("CHARMM-GUI lipid composition strings must contain at least one lipid entry.")
    return composition


def _safe_extract_tarball(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            member_path = (destination / member.name).resolve()
            if os.path.commonpath([str(destination.resolve()), str(member_path)]) != str(destination.resolve()):
                raise ValueError(f"Refusing to extract unsafe tar member outside target directory: {member.name}")
        archive.extractall(destination)


def _select_best_path(candidates: Iterable[Path], *, directory_hint: Optional[Path] = None, stem_hint: Optional[str] = None) -> Optional[Path]:
    candidates = [candidate for candidate in candidates if candidate.is_file()]
    if not candidates:
        return None

    def rank(path: Path) -> tuple[int, int, int, int, str]:
        path_parts = {part.lower() for part in path.parts}
        in_amber_dir = 0 if "amber" in path_parts else 1
        same_directory = 0 if directory_hint is not None and path.parent == directory_hint else 1
        preferred_name = path.name.lower()
        stem_priority = 2
        if stem_hint and path.stem == stem_hint:
            stem_priority = 0
        elif "step5_input" in preferred_name:
            stem_priority = 1
        return (in_amber_dir, same_directory, stem_priority, len(path.parts), path.as_posix())

    return sorted(candidates, key=rank)[0]


def _parse_html_form_defaults(
    html_text: str,
    *,
    form_id: Optional[str] = None,
    form_name: Optional[str] = None,
) -> tuple[str, str, list[tuple[str, str]]]:
    parser = _HTMLFormParser(target_id=form_id, target_name=form_name)
    parser.feed(html_text)
    form = parser.matching_form()
    if form is None:
        raise RuntimeError("Could not locate the expected CHARMM-GUI HTML form.")

    fields: list[tuple[str, str]] = []
    for attrs in form["inputs"]:
        name = attrs.get("name")
        if not name:
            continue
        input_type = str(attrs.get("type", "text")).lower()
        if input_type in {"submit", "button", "image", "file", "reset"}:
            continue
        if input_type in {"checkbox", "radio"} and "checked" not in attrs:
            continue
        value = attrs.get("value")
        if value is None:
            value = "on" if input_type in {"checkbox", "radio"} else ""
        fields.append((str(name), str(value)))

    for select in form["selects"]:
        name = select["attrs"].get("name")
        if not name:
            continue
        chosen = None
        for option in select["options"]:
            if "selected" in option["attrs"]:
                chosen = option
                break
        if chosen is None and select["options"]:
            chosen = select["options"][0]
        if chosen is None:
            continue
        value = chosen["attrs"].get("value")
        if value is None:
            value = html.unescape(str(chosen["text"]).strip())
        fields.append((str(name), str(value)))

    action = str(form["attrs"].get("action", "") or "")
    method = str(form["attrs"].get("method", "GET") or "GET").upper()
    return (action, method, fields)


def _parse_pdb_reader_prot_chains(html_text: str) -> Dict[str, Dict[str, str]]:
    match = re.search(r"var prot_chains\s*=\s*(\{.*?\})\s*;", html_text, re.S)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}

    prot_chains: Dict[str, Dict[str, str]] = {}
    for segid, residues in parsed.items():
        if not isinstance(residues, Mapping):
            continue
        prot_chains[str(segid)] = {str(resid): str(resname) for resid, resname in residues.items()}
    return prot_chains


def _parse_glycan_yaml_entries(
    yaml_text: str,
    prot_chains: Mapping[str, Mapping[str, str]],
    *,
    input_chain_residues: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> list[Dict[str, str]]:
    entries: list[Dict[str, str]] = []
    for match in re.finditer(r"(?ms)^-\s+chain:\s*(?P<chain>\S+)\n(?P<body>.*?)(?=^-\s+chain:|\Z)", yaml_text):
        chain = str(match.group("chain")).strip()
        body = match.group("body")

        segid_match = re.search(r"(?m)^\s*segid:\s*(\S+)\s*$", body)
        glycan_type_match = re.search(r"(?m)^\s*type:\s*([^\n]+?)\s*$", body)
        linked_chain_match = re.search(r"linked_chain:\s*(\S+)", body)
        linked_resid_match = re.search(r"linked_resid:\s*(\d+)", body)
        grs_match = re.search(r"(?ms)^\s*grs:\s*\|\n(?P<grs>(?:\s{4,}[^\n]*\n?)*)", body)
        if not (segid_match and glycan_type_match and linked_chain_match and linked_resid_match and grs_match):
            continue

        linked_chain = str(linked_chain_match.group(1)).strip().rstrip(",")
        protein_segid = linked_chain if linked_chain.startswith("PRO") else f"PRO{linked_chain}"
        linked_resid = str(linked_resid_match.group(1)).strip()
        protein_resname = str(prot_chains.get(protein_segid, {}).get(linked_resid, "")).strip()
        if not protein_resname and input_chain_residues is not None:
            protein_resname = str(input_chain_residues.get(linked_chain, {}).get(linked_resid, "")).strip()
        if not protein_resname:
            continue

        residue_ids_match = re.search(r"residue_list:\s*\[(?P<ids>[^\]]+)\]", body)
        residue_ids: list[str] = []
        if residue_ids_match:
            residue_ids = [
                token.strip().strip("'\"")
                for token in residue_ids_match.group("ids").split(",")
                if token.strip()
            ]
        if not residue_ids:
            residue_ids = re.findall(r"(?<!linked_)resid:\s*([A-Za-z0-9]+)", body)

        input_chain = chain
        if input_chain_residues is not None:
            inferred_input_chain = _infer_input_glycan_chain(
                linked_chain=linked_chain,
                residue_ids=residue_ids,
                input_chain_residues=input_chain_residues,
            )
            if inferred_input_chain:
                input_chain = inferred_input_chain

        grs_lines = [line.strip() for line in grs_match.group("grs").splitlines() if line.strip()]
        if not grs_lines:
            continue

        entries.append(
            {
                "chain": chain,
                "input_chain": input_chain,
                "submission_chain": "none",
                "residue_ids": list(residue_ids),
                "segid": str(segid_match.group(1)).strip(),
                "type": str(glycan_type_match.group(1)).strip(),
                "protein_segid": protein_segid,
                "protein_resid": linked_resid,
                "protein_resname": protein_resname,
                "grs": "\n".join(grs_lines),
            }
        )

    return entries


def _parse_pdb_chain_residues(pdb_path: Path) -> Dict[str, Dict[str, str]]:
    chain_residues: Dict[str, Dict[str, str]] = {}
    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            chain_id = line[21].strip()
            resid = line[22:26].strip()
            resname = line[17:20].strip()
            if not chain_id or not resid or not resname:
                continue
            chain_residues.setdefault(chain_id, {})
            chain_residues[chain_id].setdefault(resid, resname)
    return chain_residues


def _infer_input_glycan_chain(
    *,
    linked_chain: str,
    residue_ids: Sequence[str],
    input_chain_residues: Mapping[str, Mapping[str, str]],
) -> Optional[str]:
    normalized_residue_ids = [str(resid).strip() for resid in residue_ids if str(resid).strip()]
    if not normalized_residue_ids:
        return None

    candidate_chains: list[str] = []
    for chain_id, residues in input_chain_residues.items():
        if str(chain_id).strip() == str(linked_chain).strip():
            continue
        residue_keys = {str(resid).strip() for resid in residues}
        if all(residue_id in residue_keys for residue_id in normalized_residue_ids):
            candidate_chains.append(str(chain_id).strip())

    if len(candidate_chains) == 1:
        return candidate_chains[0]
    return None


def _sanitize_pdb_reader_upload_file(upload_file: Path, model_root: Path) -> Path:
    if upload_file.suffix.lower() not in {".pdb", ".ent"}:
        return upload_file

    glycan_yaml_paths = sorted(model_root.rglob("glycan.yml"))
    if not glycan_yaml_paths:
        return upload_file

    input_chain_residues = _parse_pdb_chain_residues(upload_file)
    if not input_chain_residues:
        return upload_file

    glycan_entries: list[Dict[str, str]] = []
    for glycan_yaml_path in glycan_yaml_paths:
        try:
            yaml_text = glycan_yaml_path.read_text(encoding="utf-8")
        except OSError:
            continue
        glycan_entries.extend(
            _parse_glycan_yaml_entries(
                yaml_text,
                {},
                input_chain_residues=input_chain_residues,
            )
        )

    residues_to_remove: set[tuple[str, str]] = set()
    chains_to_drop_ter: set[str] = set()
    for entry in glycan_entries:
        input_chain = str(entry.get("input_chain", "")).strip()
        residue_ids = [str(resid).strip() for resid in entry.get("residue_ids", []) if str(resid).strip()]
        if not input_chain or not residue_ids:
            continue
        chains_to_drop_ter.add(input_chain)
        for residue_id in residue_ids:
            residues_to_remove.add((input_chain, residue_id))

    if not residues_to_remove:
        return upload_file

    sanitized_lines: list[str] = []
    changed = False
    with upload_file.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM", "ANISOU")):
                chain_id = line[21].strip()
                resid = line[22:26].strip()
                if (chain_id, resid) in residues_to_remove:
                    changed = True
                    continue
            if line.startswith("TER"):
                chain_id = line[21].strip()
                if chain_id in chains_to_drop_ter:
                    changed = True
                    continue
            if line.startswith("LINK"):
                chain_a = line[21].strip()
                resid_a = line[22:26].strip()
                chain_b = line[51].strip()
                resid_b = line[52:56].strip()
                if (chain_a, resid_a) in residues_to_remove or (chain_b, resid_b) in residues_to_remove:
                    changed = True
                    continue
            if line.startswith("CONECT"):
                atom_tokens = line.split()[1:]
                if atom_tokens:
                    changed = True
                    continue
            sanitized_lines.append(line)

    if not changed:
        return upload_file

    sanitized_path = model_root / f"{upload_file.stem}.pdbreader_upload{upload_file.suffix}"
    sanitized_path.write_text("".join(sanitized_lines), encoding="utf-8")
    return sanitized_path


def _parse_pdb_residue_sequence(pdb_path: Path) -> list[Dict[str, str]]:
    residues: list[Dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            segid = line[72:76].strip()
            resid = line[22:26].strip()
            icode = line[26].strip()
            resname = line[17:20].strip()
            key = (segid, resid, icode, resname)
            if key in seen:
                continue
            seen.add(key)
            residues.append(
                {
                    "segid": segid,
                    "resid": resid,
                    "icode": icode,
                    "resname": resname,
                }
            )
    return residues


def _normalize_protein_resname(resname: str) -> str:
    return _PROTEIN_RESIDUE_EQUIVALENTS.get(str(resname).strip().upper(), str(resname).strip().upper())


def _encode_multipart_formdata(
    fields: Sequence[tuple[str, str]],
    files: Sequence[tuple[str, str, bytes, Optional[str]]],
) -> tuple[bytes, str]:
    boundary = f"----prepare-proteins-{uuid.uuid4().hex}"
    body_parts: list[bytes] = []

    for name, value in fields:
        body_parts.append(f"--{boundary}\r\n".encode("utf-8"))
        body_parts.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8")
        )

    for name, filename, payload, content_type in files:
        safe_content_type = content_type or "application/octet-stream"
        body_parts.append(f"--{boundary}\r\n".encode("utf-8"))
        body_parts.append(
            (
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {safe_content_type}\r\n\r\n"
            ).encode("utf-8")
        )
        body_parts.append(payload)
        body_parts.append(b"\r\n")

    body_parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return (b"".join(body_parts), f"multipart/form-data; boundary={boundary}")


@register_backend
class CHARMMGUIBackend(ParameterizationBackend):
    """Parameterization backend for CHARMM-GUI Quick Bilayer and fixed-size Membrane Builder jobs."""

    name = "charmm_gui"
    input_format = "amber"

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        defaults = {
            "api_base_url": "https://www.charmm-gui.org/api",
            "site_base_url": "https://charmm-gui.org",
            "token_env": "CHARMMGUI_TOKEN",
            "email_env": "CHARMMGUI_EMAIL",
            "password_env": "CHARMMGUI_PASSWORD",
            "poll_interval_s": 30.0,
            "timeout_s": 7200.0,
            "api_request_timeout_s": 120.0,
            "site_request_timeout_s": 120.0,
            "membrane_builder_submit_timeout_s": 20.0,
            "pdb_reader_poll_interval_s": 5.0,
            "pdb_reader_timeout_s": 900.0,
            "quick_bilayer_activation_poll_interval_s": 5.0,
            "quick_bilayer_activation_timeout_s": 300.0,
            "membrane_builder_activation_poll_interval_s": 5.0,
            "membrane_builder_activation_timeout_s": 300.0,
            "verify_quick_bilayer_submission": True,
            "quick_bilayer_site_fallback": True,
            "prefer_site_download": False,
            "output_format": "amber",
        }
        for key, value in defaults.items():
            self.options.setdefault(key, value)

    def prepare_model(self, openmm_md, parameters_folder: str, **kwargs: Any) -> ParameterizationResult:
        options = dict(self.options)
        options.update(kwargs)

        model_name = getattr(openmm_md, "pdb_name", None) or "model"
        model_root = Path(parameters_folder) / model_name
        model_root.mkdir(parents=True, exist_ok=True)

        input_pdb = getattr(openmm_md, "input_pdb", None)
        if input_pdb and os.path.exists(input_pdb):
            copied_input = model_root / f"{model_name}.input.pdb"
            if not copied_input.exists():
                shutil.copyfile(input_pdb, copied_input)

        cached_result = self._load_cached_result(openmm_md, model_root)
        if cached_result is not None:
            openmm_md.parameterization_result = cached_result
            return cached_result

        workflow_mode = _normalize_workflow_mode(options.get("workflow_mode", "full"))
        allow_remote = bool(options.get("allow_remote", True))
        allow_submit = allow_remote and workflow_mode in {"full", "submit_only"}
        output_format = str(options.get("output_format", "amber")).strip().lower()
        if output_format != "amber":
            raise ValueError("CHARMM-GUI backend currently supports only output_format='amber'.")

        api_base_url = str(options.get("api_base_url", "")).rstrip("/")
        if not api_base_url:
            raise ValueError("CHARMM-GUI backend requires a non-empty api_base_url.")
        site_base_url = str(options.get("site_base_url", "")).rstrip("/")
        if not site_base_url:
            raise ValueError("CHARMM-GUI backend requires a non-empty site_base_url.")
        token = None
        cached_request_payload = self._load_cached_request_payload(model_root)

        download_path = model_root / "download.tgz"
        extract_root = model_root / "extracted"
        submit_response_path = model_root / "submit_response.json"

        submit_response = None
        submitted_now = False
        if submit_response_path.exists():
            submit_response = self._read_json(submit_response_path)
        elif not allow_submit:
            raise FileNotFoundError(
                f"No cached CHARMM-GUI submission metadata found for model {model_name} in {model_root} "
                "and remote submission is disabled."
            )

        request_mode = self._resolve_request_mode(
            options=options,
            model_name=model_name,
            cached_request_payload=cached_request_payload,
            submit_response=submit_response,
            workflow_mode=workflow_mode,
        )
        request_payload = None
        membrane_builder = None
        source_jobid_override = ""
        if request_mode == "membrane_builder":
            membrane_builder = _resolve_model_option(
                options.get("membrane_builder"),
                model_name,
                _MEMBRANE_BUILDER_OPTION_KEYS,
            )
            if membrane_builder is not None and not isinstance(membrane_builder, Mapping):
                raise ValueError("CHARMM-GUI backend expects 'membrane_builder' to be a mapping when provided.")
            if isinstance(cached_request_payload, Mapping):
                query = cached_request_payload.get("query", {})
                if isinstance(query, Mapping):
                    source_jobid_override = str(query.get("jobid", "")).strip()
            if isinstance(membrane_builder, Mapping):
                source_jobid_override = str(
                    membrane_builder.get("source_jobid", source_jobid_override)
                ).strip() or source_jobid_override

        pdb_reader_jobid = source_jobid_override or self._resolve_or_create_pdb_reader_jobid(
            openmm_md=openmm_md,
            model_root=model_root,
            options=options,
            model_name=model_name,
            allow_remote=allow_submit,
        )

        if request_mode == "membrane_builder":
            step1_snapshot_path = model_root / "membrane_builder_step1.html"
            submit_snapshot_path = model_root / "membrane_builder_submit.html"
            retriever_snapshot_path = model_root / "membrane_builder_retriever.html"
            jobids_snapshot_path = model_root / "membrane_builder_jobids.html"
            ffconverter_step1_snapshot_path = model_root / "ffconverter_step1.html"
            ffconverter_submit_snapshot_path = model_root / "ffconverter_submit.html"
            ffconverter_retriever_snapshot_path = model_root / "ffconverter_retriever.html"
            ffconverter_jobids_snapshot_path = model_root / "ffconverter_jobids.html"

            if cached_request_payload is not None and (
                submit_response is not None or workflow_mode == "collect" or not allow_submit
            ):
                request_payload = cached_request_payload
            elif membrane_builder is None:
                raise ValueError(
                    "CHARMM-GUI backend requires a 'membrane_builder' mapping in parameterization_options."
                )
            else:
                email, password = self._resolve_login_credentials(options, model_name)
                if not email or not password:
                    raise ValueError(
                        "CHARMM-GUI membrane_builder submissions require website login credentials. "
                        "Provide email/password (or the configured env vars)."
                    )
                session = self._create_site_session(site_base_url, email, password)
                source_project = str(membrane_builder.get("source_project", "pdbreader")).strip() or "pdbreader"
                request_payload, submit_response = self._submit_membrane_builder(
                    session=session,
                    site_base_url=site_base_url,
                    source_jobid=pdb_reader_jobid,
                    source_project=source_project,
                    membrane_builder=dict(membrane_builder),
                    step1_snapshot_path=step1_snapshot_path,
                    submit_snapshot_path=submit_snapshot_path,
                )
                self._write_json(model_root / "request.json", request_payload)
                self._write_json(submit_response_path, submit_response)
                submitted_now = True

            if submit_response is None:
                raise FileNotFoundError(
                    f"No cached CHARMM-GUI membrane_builder submission metadata found for model {model_name} in "
                    f"{model_root} and remote submission is disabled."
                )

            if submitted_now:
                submission_epoch = int(time.time())
            else:
                submission_epoch = (
                    int(submit_response_path.stat().st_mtime) if submit_response_path.exists() else int(time.time())
                )

            membrane_builder_jobid = str(submit_response.get("jobid", "")).strip() or str(pdb_reader_jobid).strip()
            membrane_builder_jobid = self._resolve_membrane_builder_jobid(
                model_root=model_root,
                options=options,
                model_name=model_name,
                candidate_jobid=membrane_builder_jobid,
                pdb_reader_jobid=pdb_reader_jobid,
                submitted_now=submitted_now,
                site_base_url=site_base_url,
                submission_epoch=submission_epoch,
                retriever_snapshot_path=retriever_snapshot_path,
                jobids_snapshot_path=jobids_snapshot_path,
            )

            if workflow_mode == "submit_only":
                result = self._build_pending_result(
                    pdb_reader_jobid=pdb_reader_jobid,
                    membrane_builder_jobid=membrane_builder_jobid,
                    request_payload=request_payload,
                    module="membrane_builder",
                )
                openmm_md.parameterization_result = result
                return result

            email, password = self._resolve_login_credentials(options, model_name)
            if not email or not password:
                raise ValueError(
                    "CHARMM-GUI membrane_builder collection requires website login credentials. "
                    "Provide email/password (or the configured env vars)."
                )
            session = self._create_site_session(site_base_url, email, password)
            membrane_builder_status = self._advance_membrane_builder_to_ffconverter(
                session=session,
                site_base_url=site_base_url,
                model_root=model_root,
                membrane_builder_jobid=membrane_builder_jobid,
                source_project=str(request_payload.get("query", {}).get("project", "pdbreader")).strip() or "pdbreader",
                retriever_snapshot_path=retriever_snapshot_path,
                jobids_snapshot_path=jobids_snapshot_path,
                progress_snapshot_path=submit_snapshot_path,
            )
            ffconverter_step = membrane_builder_status.get("ffconverter_step")
            if not isinstance(ffconverter_step, Mapping):
                result = self._build_pending_result(
                    pdb_reader_jobid=pdb_reader_jobid,
                    membrane_builder_jobid=membrane_builder_jobid,
                    request_payload=request_payload,
                    state=str(membrane_builder_status.get("state", "membrane_builder_pending")),
                    module="membrane_builder",
                    extra_metadata={key: value for key, value in membrane_builder_status.items() if key != "ffconverter_step"},
                )
                openmm_md.parameterization_result = result
                return result

            ffconverter_status = self._ensure_ffconverter_submitted(
                session=session,
                site_base_url=site_base_url,
                model_root=model_root,
                ffconverter_step=ffconverter_step,
                step1_snapshot_path=ffconverter_step1_snapshot_path,
                submit_snapshot_path=ffconverter_submit_snapshot_path,
                retriever_snapshot_path=ffconverter_retriever_snapshot_path,
            )
            ffconverter_jobid = str(ffconverter_status.get("jobid", "")).strip() or membrane_builder_jobid
            if not _normalize_bool(ffconverter_status.get("ready_for_polling"), default=False):
                result = self._build_pending_result(
                    pdb_reader_jobid=pdb_reader_jobid,
                    membrane_builder_jobid=membrane_builder_jobid,
                    ffconverter_jobid=ffconverter_jobid,
                    request_payload=request_payload,
                    state=str(ffconverter_status.get("state", "ffconverter_pending")),
                    module="membrane_builder",
                    extra_metadata={key: value for key, value in ffconverter_status.items() if key != "ready_for_polling"},
                )
                openmm_md.parameterization_result = result
                return result

            token = self._resolve_token(options, model_name, api_base_url=api_base_url)
            final_status = self._poll_until_done(
                api_base_url,
                token,
                ffconverter_jobid,
                status_history_path=model_root / "status_history.jsonl",
                poll_interval_s=float(options.get("poll_interval_s", 30.0)),
                timeout_s=float(options.get("timeout_s", 7200.0)),
                site_base_url=site_base_url,
                email=email,
                password=password,
                jobids_snapshot_path=ffconverter_jobids_snapshot_path,
                job_modules=("Force Field Converter", "Quick Bilayer"),
            )
            self._write_json(model_root / "final_status.json", final_status)
            self._download_archive(
                api_base_url,
                token,
                ffconverter_jobid,
                download_path,
                site_base_url=site_base_url,
                email=email,
                password=password,
                poll_interval_s=float(options.get("poll_interval_s", 30.0)),
                timeout_s=float(options.get("timeout_s", 7200.0)),
            )
            if extract_root.exists():
                shutil.rmtree(extract_root)
            _safe_extract_tarball(download_path, extract_root)
            if not self._extracted_tree_has_amber_inputs(extract_root):
                result = self._build_pending_result(
                    pdb_reader_jobid=pdb_reader_jobid,
                    membrane_builder_jobid=membrane_builder_jobid,
                    ffconverter_jobid=ffconverter_jobid,
                    request_payload=request_payload,
                    state="download_incomplete",
                    module="membrane_builder",
                    extra_metadata={
                        "archive_path": str(download_path),
                        "reason": "downloaded archive does not yet contain AMBER inputs",
                        "final_status": final_status,
                    },
                )
                openmm_md.parameterization_result = result
                return result

            result = self._build_result_from_extracted_tree(
                openmm_md=openmm_md,
                model_root=model_root,
                extract_root=extract_root,
                pdb_reader_jobid=pdb_reader_jobid,
                membrane_builder_jobid=membrane_builder_jobid,
                ffconverter_jobid=ffconverter_jobid,
                request_payload=request_payload,
            )
            openmm_md.parameterization_result = result
            return result

        quick_page_snapshot_path = model_root / "quick_bilayer_page.html"
        retriever_snapshot_path = model_root / "quick_bilayer_retriever.html"
        jobids_snapshot_path = model_root / "quick_bilayer_jobids.html"

        if cached_request_payload is not None and (
            submit_response is not None or workflow_mode == "collect" or not allow_submit
        ):
            request_payload = cached_request_payload
        else:
            quick_bilayer = _resolve_model_option(options.get("quick_bilayer"), model_name, _QUICK_BILAYER_OPTION_KEYS)
            if not isinstance(quick_bilayer, Mapping):
                raise ValueError("CHARMM-GUI backend requires a 'quick_bilayer' mapping in parameterization_options.")
            quick_bilayer = dict(quick_bilayer)
            request_payload = self._build_quick_bilayer_request(pdb_reader_jobid, quick_bilayer)
            self._write_json(model_root / "request.json", request_payload)
        request_query = request_payload.get("query", {}) if isinstance(request_payload, Mapping) else {}
        require_ffconverter = _normalize_bool(
            request_query.get("run_ffconverter") if isinstance(request_query, Mapping) else None,
            default=True,
        )

        if submit_response is None:
            token = self._resolve_token(options, model_name, api_base_url=api_base_url)
            try:
                submit_response = self._submit_quick_bilayer(api_base_url, token, request_payload)
            except RuntimeError as exc:
                allow_site_fallback = _normalize_bool(options.get("quick_bilayer_site_fallback"), default=True)
                email, password = self._resolve_login_credentials(options, model_name)
                if not allow_site_fallback or not email or not password:
                    raise
                session = self._create_site_session(site_base_url, email, password)
                submit_response = self._submit_quick_bilayer_via_site(
                    session=session,
                    site_base_url=site_base_url,
                    request_payload=request_payload,
                    submit_snapshot_path=quick_page_snapshot_path,
                )
            self._write_json(submit_response_path, submit_response)
            submitted_now = True
        if submitted_now:
            submission_epoch = int(time.time())
        else:
            submission_epoch = (
                int(submit_response_path.stat().st_mtime) if submit_response_path.exists() else int(time.time())
            )

        quick_bilayer_jobid = str(submit_response.get("jobid", "")).strip()
        if not quick_bilayer_jobid:
            raise RuntimeError("CHARMM-GUI Quick Bilayer response did not include a jobid.")
        quick_bilayer_jobid = self._resolve_quick_bilayer_jobid(
            model_root=model_root,
            options=options,
            model_name=model_name,
            candidate_jobid=quick_bilayer_jobid,
            pdb_reader_jobid=pdb_reader_jobid,
            submitted_now=submitted_now,
            workflow_mode=workflow_mode,
            site_base_url=site_base_url,
            submission_epoch=submission_epoch,
            quick_page_snapshot_path=quick_page_snapshot_path,
            jobids_snapshot_path=jobids_snapshot_path,
        )

        if workflow_mode == "submit_only":
            result = self._build_pending_result(
                pdb_reader_jobid=pdb_reader_jobid,
                quick_bilayer_jobid=quick_bilayer_jobid,
                request_payload=request_payload,
                module="quick_bilayer",
            )
            openmm_md.parameterization_result = result
            return result

        archive_ready = download_path.exists()
        if archive_ready:
            if extract_root.exists():
                shutil.rmtree(extract_root)
            try:
                _safe_extract_tarball(download_path, extract_root)
            except (tarfile.TarError, EOFError, OSError):
                if extract_root.exists():
                    shutil.rmtree(extract_root)
                download_path.unlink(missing_ok=True)
                archive_ready = False
            else:
                if not self._extracted_tree_has_amber_inputs(extract_root):
                    if extract_root.exists():
                        shutil.rmtree(extract_root)
                    if allow_remote:
                        download_path.unlink(missing_ok=True)
                        archive_ready = False
                    else:
                        raise FileNotFoundError(
                            f"Cached CHARMM-GUI archive for model {model_name} in {download_path} does not contain "
                            "AMBER inputs. Rerun with remote access enabled to refresh the archive."
                        )

        final_status = None
        if not archive_ready:
            if not allow_remote:
                raise FileNotFoundError(
                    f"CHARMM-GUI archive for model {model_name} was not found at {download_path} "
                    "and remote download is disabled."
                )
            if token is None:
                token = self._resolve_token(options, model_name, api_base_url=api_base_url)
            email, password = self._resolve_login_credentials(options, model_name)
            final_status = self._poll_until_done(
                api_base_url,
                token,
                quick_bilayer_jobid,
                status_history_path=model_root / "status_history.jsonl",
                poll_interval_s=float(options.get("poll_interval_s", 30.0)),
                timeout_s=float(options.get("timeout_s", 7200.0)),
                site_base_url=site_base_url if email and password else None,
                email=email,
                password=password,
                jobids_snapshot_path=jobids_snapshot_path,
            )
            self._write_json(model_root / "final_status.json", final_status)
            site_status_source = str(final_status.get("source", "")).strip().lower()
            prefer_site_download = _normalize_bool(options.get("prefer_site_download"), default=False)
            if require_ffconverter and email and password and site_status_source == "jobids_page":
                retriever_status = self._get_live_retriever_status(
                    site_base_url=site_base_url,
                    email=email,
                    password=password,
                    jobid=quick_bilayer_jobid,
                    retriever_snapshot_path=retriever_snapshot_path,
                )
                ffconverter_step = retriever_status.get("ffconverter_step")
                if not isinstance(ffconverter_step, Mapping):
                    pending_result = self._build_pending_result(
                        pdb_reader_jobid=pdb_reader_jobid,
                        quick_bilayer_jobid=quick_bilayer_jobid,
                        request_payload=request_payload,
                        state="ffconverter_pending",
                        extra_metadata={
                            "reason": "FF-Converter step is not yet exposed in the live retriever page",
                            "final_status": final_status,
                            "retriever_steps": retriever_status.get("steps", []),
                        },
                    )
                    openmm_md.parameterization_result = pending_result
                    return pending_result
            self._download_archive(
                api_base_url,
                token,
                quick_bilayer_jobid,
                download_path,
                site_base_url=(
                    site_base_url
                    if email and password and (prefer_site_download or site_status_source == "jobids_page")
                    else None
                ),
                email=email,
                password=password,
                poll_interval_s=float(options.get("poll_interval_s", 30.0)),
                timeout_s=float(options.get("timeout_s", 7200.0)),
            )
            if extract_root.exists():
                shutil.rmtree(extract_root)
            _safe_extract_tarball(download_path, extract_root)
            if not self._extracted_tree_has_amber_inputs(extract_root):
                pending_result = self._build_pending_result(
                    pdb_reader_jobid=pdb_reader_jobid,
                    quick_bilayer_jobid=quick_bilayer_jobid,
                    request_payload=request_payload,
                    state="download_incomplete",
                    extra_metadata={
                        "archive_path": str(download_path),
                        "reason": "downloaded archive does not yet contain AMBER inputs",
                        "final_status": final_status,
                    },
                )
                openmm_md.parameterization_result = pending_result
                return pending_result

        result = self._build_result_from_extracted_tree(
            openmm_md=openmm_md,
            model_root=model_root,
            extract_root=extract_root,
            pdb_reader_jobid=pdb_reader_jobid,
            quick_bilayer_jobid=quick_bilayer_jobid,
            request_payload=request_payload,
        )
        openmm_md.parameterization_result = result
        return result

    def describe_model(self, openmm_md) -> ParameterizationResult:
        cached = getattr(openmm_md, "parameterization_result", None)
        if isinstance(cached, ParameterizationResult):
            return cached
        return super().describe_model(openmm_md)

    def _resolve_or_create_pdb_reader_jobid(
        self,
        *,
        openmm_md,
        model_root: Path,
        options: Mapping[str, Any],
        model_name: str,
        allow_remote: bool,
    ) -> str:
        explicit_jobid = _resolve_model_option(options.get("pdb_reader_jobid"), model_name)
        if explicit_jobid is not None:
            explicit_text = str(explicit_jobid).strip()
            if not explicit_text:
                raise ValueError("CHARMM-GUI backend received an empty pdb_reader_jobid.")
            return explicit_text

        pdb_reader_config = self._resolve_pdb_reader_config(options, model_name)
        config_jobid = pdb_reader_config.get("jobid")
        if config_jobid is not None:
            config_text = str(config_jobid).strip()
            if not config_text:
                raise ValueError("CHARMM-GUI backend received an empty pdb_reader['jobid'] value.")
            return config_text

        cached_jobid = self._load_cached_pdb_reader_jobid(model_root)
        if cached_jobid is not None:
            return cached_jobid

        if not allow_remote:
            raise FileNotFoundError(
                f"No cached CHARMM-GUI artifacts found for model {model_name} in {model_root} "
                "and automatic PDB Reader creation is disabled."
            )

        return self._create_pdb_reader_job(
            openmm_md=openmm_md,
            model_root=model_root,
            options=options,
            model_name=model_name,
            pdb_reader_config=pdb_reader_config,
        )

    def _resolve_pdb_reader_config(self, options: Mapping[str, Any], model_name: str) -> Dict[str, Any]:
        pdb_reader = _resolve_model_option(options.get("pdb_reader"), model_name, _PDB_READER_OPTION_KEYS)
        if pdb_reader is None:
            return {}
        if not isinstance(pdb_reader, Mapping):
            raise ValueError("CHARMM-GUI backend expects 'pdb_reader' to be a mapping when provided.")
        return dict(pdb_reader)

    def _create_pdb_reader_job(
        self,
        *,
        openmm_md,
        model_root: Path,
        options: Mapping[str, Any],
        model_name: str,
        pdb_reader_config: Mapping[str, Any],
    ) -> str:
        site_base_url = str(options.get("site_base_url", "")).rstrip("/")
        if not site_base_url:
            raise ValueError("CHARMM-GUI backend requires a non-empty site_base_url.")

        email, password = self._resolve_login_credentials(options, model_name)
        if not email or not password:
            raise ValueError(
                "Automatic CHARMM-GUI PDB Reader creation requires login credentials. "
                "Provide parameterization_options['email'] and ['password'] (or the configured env vars), "
                "or pass an existing pdb_reader_jobid."
            )

        session = self._create_site_session(site_base_url, email, password)
        start_html = self._submit_pdb_reader_start(
            session=session,
            site_base_url=site_base_url,
            openmm_md=openmm_md,
            pdb_reader_config=pdb_reader_config,
            model_root=model_root,
        )
        self._write_text(model_root / "pdb_reader_step1_chain_selection.html", start_html)
        pdb_reader_jobid = self._extract_jobid_from_html(start_html)

        action, method, chain_fields = _parse_html_form_defaults(start_html, form_id="fpdbreader")
        filtered_chain_fields = self._filter_pdb_reader_chain_fields(chain_fields, pdb_reader_config)
        chain_selection_html = self._site_submit_form(
            session,
            site_base_url,
            action=action,
            method=method,
            fields=filtered_chain_fields,
        )
        self._write_text(model_root / "pdb_reader_step2_options.html", chain_selection_html)

        action, method, manipulation_fields = _parse_html_form_defaults(chain_selection_html, form_id="fpdbreader")
        updated_manipulation_fields = self._apply_pdb_reader_manipulation_overrides(
            manipulation_fields,
            pdb_reader_config,
            html_text=chain_selection_html,
            model_root=model_root,
        )
        try:
            reader_html = self._site_submit_form(
                session,
                site_base_url,
                action=action,
                method=method,
                fields=updated_manipulation_fields,
            )
        except TimeoutError:
            reader_html = self._recover_pdb_reader_submit_timeout(
                session=session,
                site_base_url=site_base_url,
                jobid=pdb_reader_jobid,
                poll_interval_s=float(options.get("pdb_reader_poll_interval_s", 5.0)),
                timeout_s=float(options.get("pdb_reader_timeout_s", 900.0)),
            )
        ready_html = self._poll_until_pdb_reader_ready(
            session=session,
            site_base_url=site_base_url,
            jobid=pdb_reader_jobid,
            initial_html=reader_html,
            poll_interval_s=float(options.get("pdb_reader_poll_interval_s", 5.0)),
            timeout_s=float(options.get("pdb_reader_timeout_s", 900.0)),
        )
        self._write_text(model_root / "pdb_reader_step3_reader.html", ready_html)
        self._write_json(
            model_root / "pdb_reader_manifest.json",
            {
                "jobid": pdb_reader_jobid,
                "config": dict(pdb_reader_config),
                "site_base_url": site_base_url,
            },
        )
        return pdb_reader_jobid

    def _create_site_session(self, site_base_url: str, email: str, password: str):
        cookie_jar = http.cookiejar.CookieJar()
        opener = _build_opener(urllib_request.HTTPCookieProcessor(cookie_jar))
        login_fields = [
            ("do", "login"),
            ("email", email),
            ("password", password),
        ]
        login_payload = urllib_parse.urlencode(login_fields).encode("utf-8")
        self._site_request(
            opener,
            f"{site_base_url}/?doc=sign",
            method="POST",
            data=login_payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        return opener

    def _submit_pdb_reader_start(
        self,
        *,
        session,
        site_base_url: str,
        openmm_md,
        pdb_reader_config: Mapping[str, Any],
        model_root: Optional[Path] = None,
    ) -> str:
        pdb_id = pdb_reader_config.get("pdb_id")
        upload_path = pdb_reader_config.get("upload_path")
        if upload_path is None and pdb_id is None:
            upload_path = getattr(openmm_md, "input_pdb", None)
        if pdb_id and upload_path:
            raise ValueError("CHARMM-GUI pdb_reader accepts either 'pdb_id' or 'upload_path', but not both.")

        pdb_format = str(pdb_reader_config.get("pdb_format") or "").strip()
        correct_pdb = _normalize_bool(pdb_reader_config.get("correct_pdb"), default=True)
        fields: list[tuple[str, str]] = [
            ("jobid", ""),
            ("project", "pdbreader"),
        ]
        files: list[tuple[str, str, bytes, Optional[str]]] = []

        if pdb_id:
            source = str(pdb_reader_config.get("source", "RCSB")).strip() or "RCSB"
            requested_format = pdb_format or "PDB"
            fields.extend(
                [
                    ("pdb_id", str(pdb_id).strip()),
                    ("source", source),
                    ("pdb_format", requested_format),
                ]
            )
            if correct_pdb and requested_format.lower() != "mmcif":
                fields.append(("correct_pdb_checked", "1"))
            payload = urllib_parse.urlencode(fields).encode("utf-8")
            return self._site_request(
                session,
                f"{site_base_url}/?doc=input/pdbreader&step=1",
                method="POST",
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            ).decode("utf-8", errors="replace")

        if not upload_path:
            raise ValueError(
                "CHARMM-GUI backend needs either 'pdb_reader_jobid', 'pdb_reader.jobid', "
                "'pdb_reader.pdb_id', or a local input_pdb/upload_path for automatic PDB Reader creation."
            )

        upload_file = Path(os.fspath(upload_path)).expanduser().resolve()
        if not upload_file.exists():
            raise FileNotFoundError(f"CHARMM-GUI pdb_reader upload file not found: {upload_file}")
        if model_root is not None:
            upload_file = _sanitize_pdb_reader_upload_file(upload_file, model_root)
        requested_format = pdb_format or self._infer_pdb_format(upload_file)
        fields.append(("pdb_format", requested_format))
        if correct_pdb and requested_format.lower() != "mmcif":
            fields.append(("correct_pdb_checked", "1"))
        content_type = mimetypes.guess_type(upload_file.name)[0] or "application/octet-stream"
        files.append(("file", upload_file.name, upload_file.read_bytes(), content_type))
        payload, multipart_content_type = _encode_multipart_formdata(fields, files)
        return self._site_request(
            session,
            f"{site_base_url}/?doc=input/pdbreader&step=1",
            method="POST",
            data=payload,
            headers={"Content-Type": multipart_content_type},
        ).decode("utf-8", errors="replace")

    def _filter_pdb_reader_chain_fields(
        self,
        fields: Sequence[tuple[str, str]],
        pdb_reader_config: Mapping[str, Any],
    ) -> list[tuple[str, str]]:
        include_hetero = _normalize_bool(pdb_reader_config.get("include_hetero"), default=False)
        include_water = _normalize_bool(pdb_reader_config.get("include_water"), default=False)
        include_dna = _normalize_bool(pdb_reader_config.get("include_dna"), default=False)
        include_rna = _normalize_bool(pdb_reader_config.get("include_rna"), default=False)

        selected_chain_ids = set()
        for name, _ in fields:
            match = _CHAIN_FIELD_PATTERN.match(name)
            if not match:
                continue
            chain_id, field_name = match.groups()
            if field_name != "checked":
                continue
            if chain_id.startswith("PRO"):
                selected_chain_ids.add(chain_id)
            elif chain_id.startswith("DNA") and include_dna:
                selected_chain_ids.add(chain_id)
            elif chain_id.startswith("RNA") and include_rna:
                selected_chain_ids.add(chain_id)
            elif chain_id.startswith("HET") and include_hetero:
                selected_chain_ids.add(chain_id)
            elif chain_id.startswith("WAT") and include_water:
                selected_chain_ids.add(chain_id)

        if not selected_chain_ids:
            raise ValueError(
                "Automatic CHARMM-GUI PDB Reader setup did not select any chains. "
                "Enable include_hetero/include_water/include_dna/include_rna if needed."
            )

        filtered_fields: list[tuple[str, str]] = []
        for name, value in fields:
            match = _CHAIN_FIELD_PATTERN.match(name)
            if match and match.group(1) not in selected_chain_ids:
                continue
            filtered_fields.append((name, value))
        return filtered_fields

    def _apply_pdb_reader_manipulation_overrides(
        self,
        fields: Sequence[tuple[str, str]],
        pdb_reader_config: Mapping[str, Any],
        *,
        html_text: Optional[str] = None,
        model_root: Optional[Path] = None,
    ) -> list[tuple[str, str]]:
        field_map: Dict[str, list[str]] = {}
        order: list[str] = []
        for name, value in fields:
            if name not in field_map:
                field_map[name] = []
                order.append(name)
            field_map[name].append(value)

        system_pH = pdb_reader_config.get("system_pH")
        if system_pH is not None:
            field_map["system_pH"] = [str(system_pH)]
            if "system_pH" not in order:
                order.append("system_pH")

        preserve_hydrogens = _normalize_bool(pdb_reader_config.get("preserve_hydrogens"), default=False)
        if preserve_hydrogens:
            field_map["hbuild_checked"] = ["on"]
            if "hbuild_checked" not in order:
                order.append("hbuild_checked")
        elif "hbuild_checked" in field_map:
            del field_map["hbuild_checked"]
            order = [name for name in order if name != "hbuild_checked"]

        mutation_entries = _build_mutation_entries(pdb_reader_config.get("mutations"))
        if mutation_entries:
            field_map["mutation_checked"] = ["1"]
            if "mutation_checked" not in order:
                order.append("mutation_checked")
            for field_name, entry_key in (
                ("mutation[chain][]", "chain"),
                ("mutation[rid][]", "rid"),
                ("mutation[patch][]", "patch"),
            ):
                field_map[field_name] = [entry[entry_key] for entry in mutation_entries]
                if field_name not in order:
                    order.append(field_name)

        glycan_entries: list[Dict[str, str]] = []
        if html_text and _normalize_bool(pdb_reader_config.get("include_hetero"), default=False):
            prot_chains = _parse_pdb_reader_prot_chains(html_text)
            glycan_yaml_paths: list[Path] = []
            input_chain_residues: Dict[str, Dict[str, str]] = {}
            if model_root is not None and model_root.exists():
                glycan_yaml_paths = sorted(model_root.rglob("glycan.yml"))
                input_pdb_paths = sorted(model_root.glob("*.input.pdb"))
                for input_pdb_path in input_pdb_paths:
                    try:
                        input_chain_residues = _parse_pdb_chain_residues(input_pdb_path)
                    except OSError:
                        continue
                    if input_chain_residues:
                        break
            for glycan_yaml_path in glycan_yaml_paths:
                try:
                    glycan_yaml_text = glycan_yaml_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                glycan_entries = _parse_glycan_yaml_entries(
                    glycan_yaml_text,
                    prot_chains,
                    input_chain_residues=input_chain_residues,
                )
                if glycan_entries:
                    break

        if glycan_entries:
            field_map["glyc_checked"] = ["1"]
            if "glyc_checked" not in order:
                order.append("glyc_checked")

            for field_name in ("rename[NAG]", "newname[NAG]"):
                if field_name in field_map:
                    del field_map[field_name]
                    order = [name for name in order if name != field_name]

            for glycan_entry in glycan_entries:
                segid = glycan_entry["segid"]
                glycan_fields = {
                    f"glycan[{segid}][chain]": glycan_entry.get("submission_chain", glycan_entry["chain"]),
                    f"glycan[{segid}][type]": glycan_entry["type"],
                    f"glycan[{segid}][prot]": json.dumps(
                        {
                            "segid": glycan_entry["protein_segid"],
                            "resname": glycan_entry["protein_resname"],
                            "resid": glycan_entry["protein_resid"],
                        }
                    ),
                    f"glycan[{segid}][grs]": glycan_entry["grs"],
                }
                for field_name, field_value in glycan_fields.items():
                    field_map[field_name] = [field_value]
                    if field_name not in order:
                        order.append(field_name)

        updated_fields: list[tuple[str, str]] = []
        for name in order:
            for value in field_map.get(name, []):
                updated_fields.append((name, value))
        return updated_fields

    def _recover_pdb_reader_submit_timeout(
        self,
        *,
        session,
        site_base_url: str,
        jobid: str,
        poll_interval_s: float,
        timeout_s: float,
    ) -> str:
        start_time = time.monotonic()
        while True:
            html_text = self._site_request(
                session,
                f"{site_base_url}/?doc=input/pdbreader&jobid={jobid}&project=pdbreader",
                method="GET",
            ).decode("utf-8", errors="replace")
            state = self._classify_pdb_reader_page(html_text)
            if state in {"running", "ready", "error"}:
                return html_text
            if time.monotonic() - start_time >= timeout_s:
                raise TimeoutError(
                    f"Timed out while waiting to recover the CHARMM-GUI PDB Reader step-3 response for job {jobid}."
                )
            time.sleep(poll_interval_s)

    def _poll_until_pdb_reader_ready(
        self,
        *,
        session,
        site_base_url: str,
        jobid: str,
        initial_html: str,
        poll_interval_s: float,
        timeout_s: float,
    ) -> str:
        start_time = time.monotonic()
        html_text = initial_html
        while True:
            state = self._classify_pdb_reader_page(html_text)
            if state == "ready":
                return html_text
            if state == "error":
                raise RuntimeError(
                    f"CHARMM-GUI PDB Reader job {jobid} did not finish cleanly. "
                    "The generated page indicates a PDB Reader error; check the cached HTML snapshots."
                )
            if time.monotonic() - start_time >= timeout_s:
                raise TimeoutError(f"Timed out while waiting for CHARMM-GUI PDB Reader job {jobid} to finish.")
            time.sleep(poll_interval_s)
            html_text = self._site_request(
                session,
                f"{site_base_url}/?doc=input/pdbreader&jobid={jobid}&project=pdbreader",
                method="GET",
            ).decode("utf-8", errors="replace")

    def _classify_pdb_reader_page(self, html_text: str) -> str:
        normalized = html_text.lower()
        if "mismatch in ligand atom order" in normalized:
            return "error"
        if "terminated abnormally" in normalized:
            return "error"
        if "var step = 'reader';" in html_text or 'var step = "reader";' in html_text:
            return "ready"
        return "running"

    def _extract_jobid_from_html(self, html_text: str) -> str:
        jobid_match = re.search(r"JOB ID:\s*(\d+)", html_text, re.IGNORECASE)
        if jobid_match:
            return jobid_match.group(1)
        hidden_match = re.search(r'name=["\']jobid["\'] value=["\']?(\d+)', html_text, re.IGNORECASE)
        if hidden_match:
            return hidden_match.group(1)
        raise RuntimeError("Could not determine the CHARMM-GUI PDB Reader job id from the returned HTML.")

    def _infer_pdb_format(self, upload_file: Path) -> str:
        suffix = upload_file.suffix.lower()
        if suffix in {".cif", ".mmcif"}:
            return "mmCIF"
        if suffix in {".crd"}:
            return "CHARMM"
        return "PDB"

    def _site_submit_form(
        self,
        session,
        site_base_url: str,
        *,
        action: str,
        method: str,
        fields: Sequence[tuple[str, str]],
        timeout_s: Optional[float] = None,
    ) -> str:
        form_method = (method or "GET").upper()
        if form_method != "POST":
            raise RuntimeError(f"Unsupported CHARMM-GUI form method {form_method!r}.")
        payload = urllib_parse.urlencode(list(fields)).encode("utf-8")
        html_bytes = self._site_request(
            session,
            urllib_parse.urljoin(f"{site_base_url}/", action),
            method="POST",
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout_s=timeout_s,
        )
        return html_bytes.decode("utf-8", errors="replace")

    def _site_request(
        self,
        session,
        url: str,
        *,
        method: str,
        data: Optional[bytes] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout_s: Optional[float] = None,
    ) -> bytes:
        response_body, _ = self._site_request_with_headers(
            session,
            url,
            method=method,
            data=data,
            headers=headers,
            timeout_s=timeout_s,
        )
        return response_body

    def _site_request_with_headers(
        self,
        session,
        url: str,
        *,
        method: str,
        data: Optional[bytes] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout_s: Optional[float] = None,
    ) -> tuple[bytes, Mapping[str, str]]:
        request_headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        if headers:
            request_headers.update({str(key): str(value) for key, value in headers.items()})
        request = urllib_request.Request(url, data=data, headers=request_headers, method=method)
        request_timeout_s = (
            float(timeout_s) if timeout_s is not None else float(self.options.get("site_request_timeout_s", 120.0))
        )
        try:
            try:
                response = session.open(request, timeout=request_timeout_s)
            except TypeError:
                response = session.open(request)
            with response:
                body = response.read()
                response_headers = getattr(response, "headers", {})
                return body, response_headers
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"CHARMM-GUI site request failed with HTTP {exc.code}: {body}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Failed to reach CHARMM-GUI site at {url}: {exc}") from exc

    def _resolve_token(self, options: Mapping[str, Any], model_name: str, *, api_base_url: str) -> str:
        token = _resolve_model_option(options.get("token"), model_name)
        if token is not None:
            token_text = str(token).strip()
            if token_text:
                return token_text

        token_env = str(options.get("token_env", "CHARMMGUI_TOKEN")).strip()
        if token_env:
            env_value = os.environ.get(token_env, "").strip()
            if env_value:
                return env_value

        email, password = self._resolve_login_credentials(options, model_name)
        if email is not None or password is not None:
            if not email or not password:
                raise ValueError(
                    "CHARMM-GUI backend login requires both email and password. "
                    "Provide parameterization_options['email'] and ['password'] or the configured env vars."
                )
            return self._login_for_token(api_base_url, email, password)

        raise ValueError(
            "CHARMM-GUI backend requires either an API token or login credentials. "
            "Set parameterization_options['token'], export "
            f"{token_env or 'CHARMMGUI_TOKEN'}, or provide email/password credentials."
        )

    def _resolve_login_credentials(self, options: Mapping[str, Any], model_name: str) -> tuple[Optional[str], Optional[str]]:
        email = _resolve_model_option(options.get("email"), model_name)
        password = _resolve_model_option(options.get("password"), model_name)
        email_text = str(email).strip() if email is not None else ""
        password_text = str(password).strip() if password is not None else ""

        if email_text or password_text:
            return (email_text or None, password_text or None)

        email_env = str(options.get("email_env", "CHARMMGUI_EMAIL")).strip()
        password_env = str(options.get("password_env", "CHARMMGUI_PASSWORD")).strip()
        email_value = os.environ.get(email_env, "").strip() if email_env else ""
        password_value = os.environ.get(password_env, "").strip() if password_env else ""
        return (email_value or None, password_value or None)

    def _resolve_request_mode(
        self,
        *,
        options: Mapping[str, Any],
        model_name: str,
        cached_request_payload: Optional[Mapping[str, Any]],
        submit_response: Optional[Mapping[str, Any]],
        workflow_mode: str,
    ) -> str:
        if workflow_mode == "collect":
            for payload in (cached_request_payload, submit_response):
                if not isinstance(payload, Mapping):
                    continue
                module_name = str(payload.get("module", "")).strip().lower()
                if module_name in {"quick_bilayer", "membrane_builder"}:
                    return module_name

        quick_bilayer = _resolve_model_option(options.get("quick_bilayer"), model_name, _QUICK_BILAYER_OPTION_KEYS)
        membrane_builder = _resolve_model_option(
            options.get("membrane_builder"),
            model_name,
            _MEMBRANE_BUILDER_OPTION_KEYS,
        )
        if quick_bilayer is not None and membrane_builder is not None:
            raise ValueError("Use either 'quick_bilayer' or 'membrane_builder', not both.")
        if membrane_builder is not None:
            return "membrane_builder"
        if quick_bilayer is not None:
            return "quick_bilayer"

        for payload in (cached_request_payload, submit_response):
            if not isinstance(payload, Mapping):
                continue
            module_name = str(payload.get("module", "")).strip().lower()
            if module_name in {"quick_bilayer", "membrane_builder"}:
                return module_name
        return "quick_bilayer"

    def _login_for_token(self, api_base_url: str, email: str, password: str) -> str:
        login_payload = json.dumps({"email": email, "password": password}).encode("utf-8")
        response = self._http_json(
            f"{api_base_url}/login",
            None,
            method="POST",
            data=login_payload,
            headers={"Content-Type": "application/json"},
        )
        token = str(response.get("token", "")).strip()
        if not token:
            raise RuntimeError("CHARMM-GUI login succeeded but no token was returned.")
        return token

    def _submit_membrane_builder(
        self,
        *,
        session,
        site_base_url: str,
        source_jobid: str,
        source_project: str,
        membrane_builder: Mapping[str, Any],
        step1_snapshot_path: Path,
        submit_snapshot_path: Path,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        step1_html = self._site_request(
            session,
            f"{site_base_url}/?doc=input/membrane.bilayer&step=1&jobid={source_jobid}"
            f"&project={urllib_parse.quote(source_project)}",
            method="GET",
        ).decode("utf-8", errors="replace")
        self._write_text(step1_snapshot_path, step1_html)
        action, method, builder_fields = _parse_html_form_defaults(step1_html, form_id="fpdbreader")
        updated_fields = self._apply_membrane_builder_overrides(builder_fields, membrane_builder)
        updated_fields = self._apply_membrane_builder_size_calculation(
            session=session,
            site_base_url=site_base_url,
            fields=updated_fields,
        )
        submit_timeout_s = float(self.options.get("membrane_builder_submit_timeout_s", 20.0))
        try:
            submit_html = self._site_submit_form(
                session,
                site_base_url,
                action=action,
                method=method,
                fields=updated_fields,
                timeout_s=submit_timeout_s,
            )
        except TimeoutError:
            submit_html = self._recover_membrane_builder_submit_timeout(
                session=session,
                site_base_url=site_base_url,
                jobid=str(source_jobid).strip(),
                project=str(source_project).strip() or "pdbreader",
                submit_snapshot_path=submit_snapshot_path,
            )
        submit_html = self._resolve_site_autosubmit_forms(
            session=session,
            site_base_url=site_base_url,
            html_text=submit_html,
            submit_timeout_s=submit_timeout_s,
        )
        self._write_text(submit_snapshot_path, submit_html)
        try:
            submitted_jobid = self._extract_jobid_from_html(submit_html)
        except RuntimeError:
            submitted_jobid = str(source_jobid).strip()

        request_payload: Dict[str, Any] = {
            "module": "membrane_builder",
            "query": {
                "jobid": str(source_jobid).strip(),
                "project": str(source_project).strip() or "pdbreader",
            },
            "action": action,
            "method": method,
            "form": {name: value for name, value in updated_fields},
        }
        submit_response = {
            "module": "membrane_builder",
            "jobid": str(submitted_jobid).strip(),
            "submitted": True,
            "source": "site_form",
            "project": str(source_project).strip() or "pdbreader",
        }
        return request_payload, submit_response

    def _apply_membrane_builder_size_calculation(
        self,
        *,
        session,
        site_base_url: str,
        fields: Sequence[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        filtered_fields = self._filter_nonempty_membrane_builder_fields(fields)
        payload_fields = list(filtered_fields)
        payload_fields.append(("area_updated", "size"))
        payload = urllib_parse.urlencode(payload_fields).encode("utf-8")
        _, response_headers = self._site_request_with_headers(
            session,
            f"{site_base_url}/?doc=input/membrane.bilayer.size",
            method="POST",
            data=payload,
            headers={
                "Accept": "*/*",
                "Content-Type": "application/x-www-form-urlencoded",
                "X-Requested-With": "XMLHttpRequest",
            },
        )
        header_json = None
        if hasattr(response_headers, "get"):
            header_json = response_headers.get("X-JSON")
        if not header_json:
            raise RuntimeError("CHARMM-GUI membrane_builder size calculation did not return an X-JSON header.")
        try:
            size_json = json.loads(str(header_json))
        except json.JSONDecodeError as exc:
            raise RuntimeError("CHARMM-GUI membrane_builder size calculation returned invalid JSON.") from exc

        field_map: Dict[str, str] = {}
        order: list[str] = []
        for name, value in fields:
            if name not in field_map:
                order.append(name)
            field_map[name] = value

        for leaflet_name in ("upper", "lower"):
            leaflet_values = size_json.get(leaflet_name, {})
            if not isinstance(leaflet_values, Mapping):
                continue
            for lipid_name, amount in leaflet_values.items():
                key = f"lipid_number[{leaflet_name}][{str(lipid_name).strip().lower()}]"
                field_map[key] = str(amount)
                if key not in order:
                    order.append(key)

        area_values = size_json.get("area", {})
        if isinstance(area_values, Mapping):
            for lipid_name, amount in area_values.items():
                key = f"lipid_number[area][{str(lipid_name).strip().lower()}]"
                field_map[key] = str(amount)
                if key not in order:
                    order.append(key)

        updated_fields: list[tuple[str, str]] = []
        for name in order:
            updated_fields.append((name, field_map[name]))
        return updated_fields

    def _filter_nonempty_membrane_builder_fields(self, fields: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
        selected_lipids: set[str] = set()
        lipid_ratio_pattern = re.compile(r"^lipid_(?:ratio|number)\[(upper|lower)\]\[([^\]]+)\]$")
        for name, value in fields:
            match = lipid_ratio_pattern.match(name)
            if match is None:
                continue
            value_text = str(value).strip()
            if not value_text:
                continue
            try:
                numeric_value = float(value_text)
            except ValueError:
                continue
            if numeric_value != 0:
                selected_lipids.add(match.group(2).strip().lower())

        filtered_fields: list[tuple[str, str]] = []
        for name, value in fields:
            if name.startswith("lipid_ratio[") or name.startswith("lipid_number["):
                keep = False
                for lipid_name in selected_lipids:
                    if f"[{lipid_name}]" in name:
                        keep = True
                        break
                if not keep:
                    continue
            filtered_fields.append((name, value))
        return filtered_fields

    def _apply_membrane_builder_overrides(
        self,
        fields: Sequence[tuple[str, str]],
        membrane_builder: Mapping[str, Any],
    ) -> list[tuple[str, str]]:
        upper = _parse_lipid_ratio_string(membrane_builder.get("upper"))
        lower = _parse_lipid_ratio_string(membrane_builder.get("lower"))

        lateral_length = membrane_builder.get("lateral_length", membrane_builder.get("hetero_lx"))
        if lateral_length is None:
            raise ValueError("membrane_builder requires 'lateral_length' or 'hetero_lx'.")
        lateral_length_text = str(float(lateral_length))

        hetero_wdist = membrane_builder.get("hetero_wdist", membrane_builder.get("wdist", 22.5))
        hetero_wdist_text = str(float(hetero_wdist))
        hetero_xy_option = str(membrane_builder.get("hetero_xy_option", "ratio")).strip() or "ratio"
        hetero_z_option = str(membrane_builder.get("hetero_z_option", "wdist")).strip() or "wdist"
        check_penetration = _normalize_bool(membrane_builder.get("check_penetration"), default=None)

        field_map: Dict[str, list[str]] = {}
        order: list[str] = []
        for name, value in fields:
            if name not in field_map:
                field_map[name] = []
                order.append(name)
            field_map[name].append(value)

        field_map["jobid"] = [str(field_map.get("jobid", [str("")])[0] or "")]
        field_map["project"] = [str(field_map.get("project", ["pdbreader"])[0] or "pdbreader")]
        field_map["lipid_option"] = ["hetero"]
        field_map["hetero_xy_option"] = [hetero_xy_option]
        field_map["hetero_lx"] = [lateral_length_text]
        field_map["hetero_z_option"] = [hetero_z_option]
        field_map["hetero_wdist"] = [hetero_wdist_text]
        for required_name in (
            "jobid",
            "project",
            "lipid_option",
            "hetero_xy_option",
            "hetero_lx",
            "hetero_z_option",
            "hetero_wdist",
        ):
            if required_name not in order:
                order.append(required_name)

        known_lipids: set[str] = set()
        for name in field_map:
            upper_match = re.match(r"^lipid_ratio\[upper\]\[([^\]]+)\]$", name)
            lower_match = re.match(r"^lipid_ratio\[lower\]\[([^\]]+)\]$", name)
            if upper_match:
                known_lipids.add(upper_match.group(1).strip().lower())
            if lower_match:
                known_lipids.add(lower_match.group(1).strip().lower())

        for leaflet, composition in (("upper", upper), ("lower", lower)):
            for lipid_name in composition:
                if lipid_name not in known_lipids:
                    raise ValueError(
                        f"CHARMM-GUI membrane_builder does not expose lipid {lipid_name!r} on the Bilayer Builder form."
                    )

        for lipid_name in known_lipids:
            upper_name = f"lipid_ratio[upper][{lipid_name}]"
            lower_name = f"lipid_ratio[lower][{lipid_name}]"
            field_map[upper_name] = ["0"]
            field_map[lower_name] = ["0"]
            if upper_name not in order:
                order.append(upper_name)
            if lower_name not in order:
                order.append(lower_name)

        for lipid_name, amount in upper.items():
            field_map[f"lipid_ratio[upper][{lipid_name}]"] = [amount]
        for lipid_name, amount in lower.items():
            field_map[f"lipid_ratio[lower][{lipid_name}]"] = [amount]

        if check_penetration is True:
            existing_value = field_map.get("check_penetration", ["1"])
            field_map["check_penetration"] = [existing_value[0] if existing_value else "1"]
            if "check_penetration" not in order:
                order.append("check_penetration")
        elif check_penetration is False and "check_penetration" in field_map:
            del field_map["check_penetration"]
            order = [name for name in order if name != "check_penetration"]

        updated_fields: list[tuple[str, str]] = []
        for name in order:
            for value in field_map.get(name, []):
                updated_fields.append((name, value))
        return updated_fields

    def _extract_formstop(self, html_text: str) -> Optional[int]:
        match = re.search(r"""formstop\s*=\s*(\d+)""", html_text, re.IGNORECASE)
        if match is None:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _extract_nav_title(self, html_text: str) -> str:
        match = re.search(r"""nav_title">([^<]+)<""", html_text, re.IGNORECASE)
        if match is None:
            return ""
        return html.unescape(match.group(1)).strip()

    def _extract_action_step(self, action: str) -> str:
        parsed = urllib_parse.urlparse(action)
        values = urllib_parse.parse_qs(parsed.query)
        step_values = values.get("step", [])
        if not step_values:
            return ""
        return str(step_values[0]).strip()

    def _load_membrane_builder_page_state(
        self,
        *,
        session,
        site_base_url: str,
        jobid: str,
        project: str,
        step: str,
        snapshot_path: Path,
    ) -> Dict[str, Any]:
        page_url = (
            f"{site_base_url}/?doc=input/membrane.bilayer&jobid={jobid}"
            f"&project={urllib_parse.quote(project)}&step={urllib_parse.quote(str(step))}"
        )
        html_text = self._site_request(session, page_url, method="GET").decode("utf-8", errors="replace")
        html_text = self._resolve_site_autosubmit_forms(
            session=session,
            site_base_url=site_base_url,
            html_text=html_text,
        )
        self._write_text(snapshot_path, html_text)
        action = ""
        method = "POST"
        fields: list[tuple[str, str]] = []
        try:
            action, method, fields = _parse_html_form_defaults(html_text, form_id="fmembrane.quick")
        except RuntimeError:
            try:
                action, method, fields = _parse_html_form_defaults(html_text, form_name="msg_form")
            except RuntimeError:
                pass
        return {
            "html": html_text,
            "page_url": page_url,
            "jobid": str(jobid).strip(),
            "project": str(project).strip(),
            "step": str(step).strip(),
            "formstop": self._extract_formstop(html_text),
            "nav_title": self._extract_nav_title(html_text),
            "action": action,
            "method": method,
            "fields": fields,
            "next_action_step": self._extract_action_step(action) if action else "",
        }

    def _resolve_site_autosubmit_forms(
        self,
        *,
        session,
        site_base_url: str,
        html_text: str,
        max_hops: int = 4,
        submit_timeout_s: Optional[float] = None,
    ) -> str:
        current_html = html_text
        for _ in range(max_hops):
            try:
                action, method, fields = _parse_html_form_defaults(current_html, form_name="msg_form")
            except RuntimeError:
                break
            current_html = self._site_submit_form(
                session,
                site_base_url,
                action=action,
                method=method,
                fields=fields,
                timeout_s=submit_timeout_s,
            )
        return current_html

    def _recover_membrane_builder_submit_timeout(
        self,
        *,
        session,
        site_base_url: str,
        jobid: str,
        project: str,
        submit_snapshot_path: Path,
    ) -> str:
        for step in ("2", "3", "4", "5"):
            try:
                html_text = self._site_request(
                    session,
                    f"{site_base_url}/?doc=input/membrane.bilayer&jobid={jobid}"
                    f"&project={urllib_parse.quote(project)}&step={step}",
                    method="GET",
                ).decode("utf-8", errors="replace")
            except Exception:
                continue
            formstop = self._extract_formstop(html_text)
            nav_title = self._extract_nav_title(html_text)
            if formstop in {0, 2} or nav_title:
                self._write_text(submit_snapshot_path, html_text)
                return html_text
        raise TimeoutError(
            f"Timed out while waiting for CHARMM-GUI Membrane Builder submit response for job {jobid}."
        )

    def _advance_membrane_builder_to_ffconverter(
        self,
        *,
        session,
        site_base_url: str,
        model_root: Path,
        membrane_builder_jobid: str,
        source_project: str,
        retriever_snapshot_path: Path,
        jobids_snapshot_path: Path,
        progress_snapshot_path: Path,
    ) -> Dict[str, Any]:
        current_step = str(self._load_cached_membrane_builder_step(model_root) or "2").strip() or "2"
        max_advances = int(self.options.get("membrane_builder_collect_advances", 4))
        submit_timeout_s = float(self.options.get("membrane_builder_submit_timeout_s", 20.0))
        last_steps: list[str] = []

        for _ in range(max_advances):
            retriever_html = self._site_request(
                session,
                f"{site_base_url}/?doc=input/retriever&jobid={membrane_builder_jobid}",
                method="GET",
            ).decode("utf-8", errors="replace")
            self._write_text(retriever_snapshot_path, retriever_html)
            parsed_steps = self._parse_retriever_steps(retriever_html)
            last_steps = [f"{step['doc']}:{step['step']}" for step in parsed_steps]

            ffconverter_step = self._extract_ffconverter_step_from_retriever(
                retriever_html,
                expected_project="membrane.bilayer",
            )
            if ffconverter_step is not None:
                return {
                    "state": "ffconverter_ready",
                    "ffconverter_step": ffconverter_step,
                    "retriever_steps": last_steps,
                }

            builder_step = self._extract_membrane_builder_step_from_retriever(retriever_html)
            if builder_step is not None:
                current_step = str(builder_step.get("step", current_step)).strip() or current_step
                self._write_json(
                    model_root / "membrane_builder_manifest.json",
                    {
                        "doc": "membrane.bilayer",
                        "jobid": str(builder_step.get("jobid", membrane_builder_jobid)).strip() or membrane_builder_jobid,
                        "project": str(builder_step.get("project", source_project)).strip() or source_project,
                        "source_jobid": str(membrane_builder_jobid).strip(),
                        "step": current_step,
                    },
                )
                builder_status = self._get_jobids_page_status(
                    session,
                    site_base_url=site_base_url,
                    jobid=str(builder_step.get("jobid", membrane_builder_jobid)).strip() or membrane_builder_jobid,
                    jobids_snapshot_path=jobids_snapshot_path,
                    job_modules=("Bilayer Builder",),
                )
                if (
                    builder_status is not None
                    and str(builder_status.get("status", "")).strip().lower() == "done"
                ):
                    direct_ffconverter_step = self._load_direct_ffconverter_step(
                        session=session,
                        site_base_url=site_base_url,
                        jobid=str(builder_step.get("jobid", membrane_builder_jobid)).strip() or membrane_builder_jobid,
                        project="membrane.bilayer",
                    )
                    if direct_ffconverter_step is not None:
                        return {
                            "state": "ffconverter_ready",
                            "ffconverter_step": direct_ffconverter_step,
                            "retriever_steps": last_steps,
                        }

            page_state = self._load_membrane_builder_page_state(
                session=session,
                site_base_url=site_base_url,
                jobid=membrane_builder_jobid,
                project=source_project,
                step=current_step,
                snapshot_path=progress_snapshot_path,
            )
            formstop = page_state.get("formstop")
            if formstop == 1:
                raise RuntimeError(
                    f"CHARMM-GUI Membrane Builder job {membrane_builder_jobid} failed while advancing "
                    f"step {current_step} ({page_state.get('nav_title', '')})."
                )
            if formstop == 2:
                return {
                    "state": "membrane_builder_running",
                    "step": current_step,
                    "nav_title": page_state.get("nav_title", ""),
                    "retriever_steps": last_steps,
                }
            action = str(page_state.get("action", "")).strip()
            fields = page_state.get("fields") or []
            if not action or not fields:
                return {
                    "state": "membrane_builder_pending",
                    "step": current_step,
                    "nav_title": page_state.get("nav_title", ""),
                    "retriever_steps": last_steps,
                }
            try:
                submit_html = self._site_submit_form(
                    session,
                    site_base_url,
                    action=action,
                    method=str(page_state.get("method", "POST")),
                    fields=list(fields),
                    timeout_s=submit_timeout_s,
                )
            except TimeoutError:
                submit_html = self._recover_membrane_builder_submit_timeout(
                    session=session,
                    site_base_url=site_base_url,
                    jobid=membrane_builder_jobid,
                    project=source_project,
                    submit_snapshot_path=progress_snapshot_path,
                )
            submit_html = self._resolve_site_autosubmit_forms(
                session=session,
                site_base_url=site_base_url,
                html_text=submit_html,
                submit_timeout_s=submit_timeout_s,
            )
            self._write_text(progress_snapshot_path, submit_html)
            submit_formstop = self._extract_formstop(submit_html)
            if submit_formstop == 1:
                raise RuntimeError(f"CHARMM-GUI Membrane Builder job {membrane_builder_jobid} failed after submit.")
            if submit_formstop == 2:
                return {
                    "state": "membrane_builder_running",
                    "step": current_step,
                    "nav_title": self._extract_nav_title(submit_html),
                    "retriever_steps": last_steps,
                }
            next_step = self._extract_action_step(str(page_state.get("action", "")))
            if next_step:
                current_step = next_step

        return {
            "state": "membrane_builder_pending",
            "step": current_step,
            "retriever_steps": last_steps,
        }

    def _load_direct_ffconverter_step(
        self,
        *,
        session,
        site_base_url: str,
        jobid: str,
        project: str,
    ) -> Optional[Dict[str, Any]]:
        page_html = self._site_request(
            session,
            f"{site_base_url}/?doc=input/converter.ffconverter&jobid={jobid}"
            f"&project={urllib_parse.quote(project)}&step=1",
            method="GET",
        ).decode("utf-8", errors="replace")
        if "Login</button>" in page_html:
            return None
        try:
            _parse_html_form_defaults(page_html, form_id="ffconverter")
        except RuntimeError:
            return None
        return {
            "doc": "converter.ffconverter",
            "jobid": str(jobid).strip(),
            "project": str(project).strip() or "membrane.bilayer",
            "step": "1",
        }

    def _apply_ffconverter_overrides(
        self,
        fields: Sequence[tuple[str, str]],
        *,
        output_format: str,
    ) -> list[tuple[str, str]]:
        if str(output_format).strip().lower() != "amber":
            raise ValueError("CHARMM-GUI FF Converter automation currently supports only output_format='amber'.")

        field_map: Dict[str, list[str]] = {}
        order: list[str] = []
        for name, value in fields:
            if name not in field_map:
                field_map[name] = []
                order.append(name)
            field_map[name].append(value)

        field_map["systype"] = ["bilayer"]
        field_map["fftype"] = ["amber"]
        for name in ("systype", "fftype"):
            if name not in order:
                order.append(name)

        updated_fields: list[tuple[str, str]] = []
        for name in order:
            for value in field_map.get(name, []):
                updated_fields.append((name, value))
        return updated_fields

    def _recover_ffconverter_submit_timeout(
        self,
        *,
        session,
        site_base_url: str,
        jobid: str,
        project: str,
        submit_snapshot_path: Path,
    ) -> str:
        for step in ("1", "2", "3", "4", "5", "6", "7"):
            try:
                html_text = self._site_request(
                    session,
                    f"{site_base_url}/?doc=input/converter.ffconverter&jobid={jobid}"
                    f"&project={urllib_parse.quote(project)}&step={step}",
                    method="GET",
                ).decode("utf-8", errors="replace")
            except Exception:
                continue
            if "Next Step" in html_text or self._extract_formstop(html_text) in {0, 2}:
                self._write_text(submit_snapshot_path, html_text)
                return html_text
        raise TimeoutError(f"Timed out while waiting for CHARMM-GUI FF Converter submit response for job {jobid}.")

    def _ensure_ffconverter_submitted(
        self,
        *,
        session,
        site_base_url: str,
        model_root: Path,
        ffconverter_step: Mapping[str, Any],
        step1_snapshot_path: Path,
        submit_snapshot_path: Path,
        retriever_snapshot_path: Path,
    ) -> Dict[str, Any]:
        ff_jobid = str(ffconverter_step.get("jobid", "")).strip()
        ff_project = str(ffconverter_step.get("project", "membrane.bilayer")).strip() or "membrane.bilayer"
        ff_step_number = str(ffconverter_step.get("step", "")).strip() or "1"
        if ff_step_number not in {"", "1"}:
            return {
                "state": "ffconverter_running",
                "jobid": ff_jobid,
                "project": ff_project,
                "step": ff_step_number,
                "ready_for_polling": True,
            }

        page_html = self._site_request(
            session,
            f"{site_base_url}/?doc=input/converter.ffconverter&jobid={ff_jobid}"
            f"&project={urllib_parse.quote(ff_project)}&step=1",
            method="GET",
        ).decode("utf-8", errors="replace")
        self._write_text(step1_snapshot_path, page_html)
        action, method, fields = _parse_html_form_defaults(page_html, form_id="ffconverter")
        fields = self._apply_ffconverter_overrides(fields, output_format="amber")
        try:
            submit_html = self._site_submit_form(
                session,
                site_base_url,
                action=action,
                method=method,
                fields=fields,
            )
        except TimeoutError:
            submit_html = self._recover_ffconverter_submit_timeout(
                session=session,
                site_base_url=site_base_url,
                jobid=ff_jobid,
                project=ff_project,
                submit_snapshot_path=submit_snapshot_path,
            )
        self._write_text(submit_snapshot_path, submit_html)

        retriever_html = self._site_request(
            session,
            f"{site_base_url}/?doc=input/retriever&jobid={ff_jobid}",
            method="GET",
        ).decode("utf-8", errors="replace")
        self._write_text(retriever_snapshot_path, retriever_html)
        live_step = self._extract_ffconverter_step_from_retriever(
            retriever_html,
            expected_project="membrane.bilayer",
        ) or {
            "jobid": ff_jobid,
            "project": ff_project,
            "step": "2",
        }
        live_step_number = str(live_step.get("step", "")).strip() or "2"
        return {
            "state": "ffconverter_running",
            "jobid": str(live_step.get("jobid", ff_jobid)).strip() or ff_jobid,
            "project": str(live_step.get("project", ff_project)).strip() or ff_project,
            "step": live_step_number,
            "ready_for_polling": live_step_number not in {"", "1"},
        }

    def _build_quick_bilayer_request(self, pdb_reader_jobid: str, quick_bilayer: Mapping[str, Any]) -> Dict[str, Any]:
        request_payload: Dict[str, Any] = {
            "module": "quick_bilayer",
            "query": {
                "jobid": str(pdb_reader_jobid),
            },
            "form": {},
        }

        membtype = quick_bilayer.get("membtype")
        upper = quick_bilayer.get("upper")
        lower = quick_bilayer.get("lower")
        if membtype is None and (upper is None or lower is None):
            raise ValueError("quick_bilayer requires either 'membtype' or both 'upper' and 'lower'.")
        if membtype is not None and (upper is not None or lower is not None):
            raise ValueError("quick_bilayer accepts either 'membtype' or ('upper', 'lower'), but not both.")

        margin = quick_bilayer.get("margin")
        if margin is None:
            raise ValueError("quick_bilayer requires 'margin'.")

        query = request_payload["query"]
        form = request_payload["form"]
        query["margin"] = str(float(margin))
        query["wdist"] = str(float(quick_bilayer.get("wdist", 22.5)))
        query["ion_conc"] = str(float(quick_bilayer.get("ion_conc", 0.15)))
        query["ion_type"] = str(quick_bilayer.get("ion_type", "NaCl")).strip()
        if not query["ion_type"]:
            raise ValueError("quick_bilayer ion_type must be non-empty.")

        if membtype is not None:
            membtype_text = str(membtype).strip()
            if not membtype_text:
                raise ValueError("quick_bilayer membtype must be non-empty.")
            form["membtype"] = membtype_text
        else:
            upper_text = str(upper).strip()
            lower_text = str(lower).strip()
            if not upper_text or not lower_text:
                raise ValueError("quick_bilayer upper/lower strings must be non-empty.")
            form["upper"] = upper_text
            form["lower"] = lower_text

        for option_name, request_name in (
            ("prot_projection_upper", "prot_projection_upper"),
            ("prot_projection_lower", "prot_projection_lower"),
        ):
            if option_name not in quick_bilayer:
                continue
            value = quick_bilayer.get(option_name)
            if value is None or str(value).strip() == "":
                continue
            query[request_name] = str(float(value))

        for option_name, request_name in (
            ("ppm", "ppm"),
            ("run_ppm", "ppm"),
            ("topologyIn", "topologyIn"),
            ("heteroatoms", "heteroatoms"),
            ("clone_job", "clone_job"),
            ("run_ffconverter", "run_ffconverter"),
        ):
            if option_name not in quick_bilayer:
                continue
            normalized = _normalize_bool(quick_bilayer.get(option_name))
            if normalized is None:
                continue
            query[request_name] = "true" if normalized else "false"

        if "run_ffconverter" not in query:
            query["run_ffconverter"] = "true"

        return request_payload

    def _submit_quick_bilayer(self, api_base_url: str, token: str, request_payload: Mapping[str, Any]) -> Dict[str, Any]:
        query = urllib_parse.urlencode(request_payload.get("query", {}))
        url = f"{api_base_url}/quick_bilayer?{query}"
        form = urllib_parse.urlencode(request_payload.get("form", {})).encode("utf-8")
        response = self._http_json(
            url,
            token,
            method="POST",
            data=form,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        submitted = str(response.get("submitted", "")).strip().lower()
        if submitted not in {"true", "1", "yes"}:
            raise RuntimeError(f"CHARMM-GUI Quick Bilayer submission failed: {response}")
        return response

    def _submit_quick_bilayer_via_site(
        self,
        *,
        session,
        site_base_url: str,
        request_payload: Mapping[str, Any],
        submit_snapshot_path: Path,
    ) -> Dict[str, Any]:
        fields: list[tuple[str, str]] = []
        query = request_payload.get("query", {})
        form = request_payload.get("form", {})
        if isinstance(query, Mapping):
            fields.extend((str(name), str(value)) for name, value in query.items())
        if isinstance(form, Mapping):
            fields.extend((str(name), str(value)) for name, value in form.items())
        payload = urllib_parse.urlencode(fields).encode("utf-8")
        html_text = self._site_request(
            session,
            f"{site_base_url}/?doc=input/membrane.quick&step=1",
            method="POST",
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        ).decode("utf-8", errors="replace")
        self._write_text(submit_snapshot_path, html_text)

        formstop = self._extract_formstop(html_text)
        if formstop == 1:
            raise RuntimeError("CHARMM-GUI Quick Bilayer site submission failed after submit.")

        query_jobid = str(query.get("jobid", "")).strip() if isinstance(query, Mapping) else ""
        submit_jobid = ""
        quick_step = None
        try:
            submit_jobid = self._extract_jobid_from_html(html_text)
        except RuntimeError:
            quick_step = self._extract_quick_bilayer_step_from_quick_page(html_text, query_jobid or "unknown")
            if quick_step is not None:
                submit_jobid = str(quick_step.get("jobid", "")).strip()
        else:
            quick_step = self._extract_quick_bilayer_step_from_quick_page(html_text, submit_jobid or query_jobid or "unknown")

        if not submit_jobid:
            raise RuntimeError("CHARMM-GUI Quick Bilayer site submission did not expose a downstream jobid.")

        response: Dict[str, Any] = {
            "jobid": submit_jobid,
            "modules": "membuilder ffconverter",
            "request": {
                **({str(name): str(value) for name, value in query.items()} if isinstance(query, Mapping) else {}),
                **({str(name): str(value) for name, value in form.items()} if isinstance(form, Mapping) else {}),
            },
            "source": "site",
            "submitted": True,
        }
        if quick_step is not None:
            response["project"] = str(quick_step.get("project", "")).strip()
            response["step"] = str(quick_step.get("step", "")).strip()
        return response

    def _poll_until_done(
        self,
        api_base_url: str,
        token: str,
        jobid: str,
        *,
        status_history_path: Path,
        poll_interval_s: float,
        timeout_s: float,
        site_base_url: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
        jobids_snapshot_path: Optional[Path] = None,
        job_modules: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        status_history_path.parent.mkdir(parents=True, exist_ok=True)
        start_time = time.monotonic()
        site_session = None
        if site_base_url and email and password:
            try:
                site_session = self._create_site_session(site_base_url, email, password)
            except Exception:
                site_session = None
        while True:
            status = self._check_status(api_base_url, token, jobid)
            with status_history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(status, sort_keys=True) + "\n")

            status_text = str(status.get("status", "")).strip().lower()
            if status_text in _DONE_STATUSES:
                return status
            if site_session is not None and jobids_snapshot_path is not None:
                try:
                    site_status = self._get_jobids_page_status(
                        site_session,
                        site_base_url=site_base_url,
                        jobid=jobid,
                        jobids_snapshot_path=jobids_snapshot_path,
                        job_modules=job_modules,
                    )
                except Exception:
                    site_status = None
                if site_status is not None:
                    site_status_text = str(site_status.get("status", "")).strip().lower()
                    if site_status_text in _DONE_STATUSES:
                        with status_history_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(site_status, sort_keys=True) + "\n")
                        return site_status
            if status_text == "compressing" and site_session is not None and site_base_url:
                try:
                    compression_progress = self._get_site_archive_compression_progress(
                        site_session,
                        site_base_url,
                        jobid,
                    )
                except Exception:
                    compression_progress = None
                if compression_progress is not None and compression_progress >= 100.0:
                    archive_ready_status = dict(status)
                    archive_ready_status["status"] = "done"
                    archive_ready_status["source"] = "site_archive_ready"
                    archive_ready_status["compression_progress"] = compression_progress
                    with status_history_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(archive_ready_status, sort_keys=True) + "\n")
                    return archive_ready_status
            if status_text and any(token in status_text for token in _ERROR_STATUS_TOKENS):
                last_output = status.get("lastOutLines") or status.get("lastOutFile") or status
                raise RuntimeError(f"CHARMM-GUI job {jobid} failed: {last_output}")

            if status_text and not any(token in status_text for token in _RUNNING_STATUS_TOKENS):
                raise RuntimeError(f"Unexpected CHARMM-GUI job status for job {jobid}: {status}")

            if time.monotonic() - start_time >= timeout_s:
                raise TimeoutError(f"Timed out while waiting for CHARMM-GUI job {jobid} to finish.")
            time.sleep(poll_interval_s)

    def _check_status(self, api_base_url: str, token: str, jobid: str) -> Dict[str, Any]:
        url = f"{api_base_url}/check_status?{urllib_parse.urlencode({'jobid': jobid})}"
        return self._http_json(url, token, method="GET")

    def _download_archive(
        self,
        api_base_url: str,
        token: str,
        jobid: str,
        destination: Path,
        *,
        site_base_url: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
        poll_interval_s: float = 30.0,
        timeout_s: float = 7200.0,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)

        site_error: Optional[Exception] = None
        if site_base_url and email and password:
            try:
                self._download_archive_via_site(
                    site_base_url,
                    email,
                    password,
                    jobid,
                    destination,
                    poll_interval_s=poll_interval_s,
                    timeout_s=timeout_s,
                )
                return
            except Exception as exc:  # pragma: no cover - exercised in fallback test via API success
                site_error = exc

        url = f"{api_base_url}/download?{urllib_parse.urlencode({'jobid': jobid})}"
        data = self._http_bytes(url, token, method="GET")
        if data.startswith(b"Still running compressing process"):
            if site_error is not None:
                raise RuntimeError(
                    f"CHARMM-GUI archive download via site flow failed for job {jobid}: {site_error}"
                ) from site_error
            raise RuntimeError(
                f"CHARMM-GUI archive for job {jobid} is still compressing. "
                "Retry later or use website login credentials for site-side download."
            )
        with destination.open("wb") as handle:
            handle.write(data)

    def _download_archive_via_site(
        self,
        site_base_url: str,
        email: str,
        password: str,
        jobid: str,
        destination: Path,
        *,
        poll_interval_s: float,
        timeout_s: float,
    ) -> None:
        session = self._create_site_session(site_base_url, email, password)
        download_url = f"{site_base_url}/?doc=input/download&jobid={jobid}"
        progress = self._get_site_archive_compression_progress(session, site_base_url, jobid)
        if progress is None or progress < 100.0:
            self._site_request(session, f"{download_url}&compress_only=1", method="GET")
            self._wait_for_site_archive_compression(
                session,
                site_base_url,
                jobid,
                poll_interval_s=poll_interval_s,
                timeout_s=timeout_s,
            )

        request = urllib_request.Request(
            download_url,
            headers={"Accept": "application/octet-stream,*/*;q=0.8"},
            method="GET",
        )
        temporary_destination = destination.with_name(f"{destination.name}.part")
        try:
            with session.open(request) as response, temporary_destination.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            temporary_destination.replace(destination)
        finally:
            if temporary_destination.exists():
                temporary_destination.unlink()

    def _wait_for_site_archive_compression(
        self,
        session,
        site_base_url: str,
        jobid: str,
        *,
        poll_interval_s: float,
        timeout_s: float,
    ) -> None:
        start_time = time.monotonic()
        while True:
            progress = self._get_site_archive_compression_progress(session, site_base_url, jobid)

            if progress is not None and progress >= 100.0:
                return

            if time.monotonic() - start_time >= timeout_s:
                raise TimeoutError(
                    f"Timed out while waiting for CHARMM-GUI archive compression for job {jobid} to finish."
                )
            time.sleep(poll_interval_s)

    def _get_site_archive_compression_progress(
        self,
        session,
        site_base_url: str,
        jobid: str,
    ) -> Optional[float]:
        check_url = f"{site_base_url}/?doc=input/download&jobid={jobid}&check_only=1"
        response_text = self._site_request(session, check_url, method="GET").decode("utf-8", errors="replace").strip()
        try:
            return float(response_text)
        except ValueError:
            return None

    def _extract_expected_glycan_metadata(self, extract_root: Path) -> Optional[Dict[str, Any]]:
        glycan_yaml_path = extract_root / "glycan.yml"
        if not glycan_yaml_path.exists():
            return None
        try:
            glycan_yaml_text = glycan_yaml_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

        segids = sorted({match.strip() for match in re.findall(r"(?m)^\s*segid:\s*(\S+)\s*$", glycan_yaml_text) if match.strip()})
        resnames = sorted(
            {
                match.strip()
                for match in re.findall(r"resname:\s*([A-Za-z0-9_]+)", glycan_yaml_text)
                if match.strip()
            }
        )
        return {
            "path": str(glycan_yaml_path),
            "segids": segids,
            "resnames": resnames,
        }

    def _pdb_contains_expected_glycans(self, pdb_path: Path, expected_glycan: Optional[Mapping[str, Any]]) -> bool:
        if expected_glycan is None:
            return True
        expected_segids = {str(segid).strip() for segid in expected_glycan.get("segids", []) if str(segid).strip()}
        expected_resnames = {str(resname).strip() for resname in expected_glycan.get("resnames", []) if str(resname).strip()}
        if not expected_segids and not expected_resnames:
            return True

        try:
            residues = _parse_pdb_residue_sequence(pdb_path)
        except OSError:
            return False

        for residue in residues:
            if residue["segid"] in expected_segids:
                return True
            if residue["resname"] in expected_resnames:
                return True
        return False

    def _protein_contiguity_error(self, topology, pdb_path: Path) -> Optional[str]:
        residue_sequence = [
            residue
            for residue in _parse_pdb_residue_sequence(pdb_path)
            if residue["segid"].startswith("PRO") and residue["resname"] in _PROTEIN_RESIDUES
        ]
        topology_residues = [residue for residue in topology.residues() if residue.name in _PROTEIN_RESIDUES]
        if len(topology_residues) != len(residue_sequence):
            return (
                "Protein residue count mismatch between topology and companion PDB "
                f"({len(topology_residues)} vs {len(residue_sequence)})."
            )

        atom_to_residue_index = {atom.index: atom.residue.index for atom in topology.atoms()}
        residue_neighbors: Dict[int, set[int]] = {}
        for atom1, atom2 in topology.bonds():
            residue_index_1 = atom_to_residue_index[atom1.index]
            residue_index_2 = atom_to_residue_index[atom2.index]
            if residue_index_1 == residue_index_2:
                continue
            residue_neighbors.setdefault(residue_index_1, set()).add(residue_index_2)
            residue_neighbors.setdefault(residue_index_2, set()).add(residue_index_1)

        missing_links: list[str] = []
        for index, (topology_residue, pdb_residue) in enumerate(zip(topology_residues, residue_sequence)):
            topology_name = _normalize_protein_resname(topology_residue.name)
            pdb_name = _normalize_protein_resname(pdb_residue["resname"])
            if topology_name != pdb_name:
                return (
                    "Protein residue order mismatch between topology and companion PDB "
                    f"at position {index + 1}: topology {topology_residue.name} {topology_residue.id}, "
                    f"PDB {pdb_residue['resname']} {pdb_residue['resid']}."
                )

        for index in range(len(topology_residues) - 1):
            current_residue = topology_residues[index]
            next_residue = topology_residues[index + 1]
            current_pdb = residue_sequence[index]
            next_pdb = residue_sequence[index + 1]
            if current_pdb["segid"] != next_pdb["segid"]:
                continue
            if next_residue.index not in residue_neighbors.get(current_residue.index, set()):
                missing_links.append(
                    f"{current_pdb['segid']}:{current_pdb['resname']}{current_pdb['resid']}"
                    f"-{next_pdb['resname']}{next_pdb['resid']}"
                )

        if missing_links:
            preview = ", ".join(missing_links[:5])
            if len(missing_links) > 5:
                preview += ", ..."
            return f"Missing peptide bond(s) in topology: {preview}"
        return None

    def _amber_validation_error(
        self,
        topology_path: Path,
        *,
        extract_root: Path,
        expected_glycan: Optional[Mapping[str, Any]],
    ) -> Optional[str]:
        companion_pdb = _select_best_path(
            extract_root.rglob("step5_input.pdb"),
            directory_hint=topology_path.parent,
            stem_hint=topology_path.stem,
        )
        if companion_pdb is None:
            if expected_glycan:
                return f"Could not locate a companion step5_input.pdb for {topology_path.name}."
            return None
        if not self._pdb_contains_expected_glycans(companion_pdb, expected_glycan):
            return "Final AMBER structure is missing the expected glycan residues."

        try:
            from openmm.app import AmberPrmtopFile
        except Exception as exc:
            return f"OpenMM AmberPrmtopFile import failed while validating CHARMM-GUI AMBER inputs: {exc}"

        topology = AmberPrmtopFile(str(topology_path)).topology
        return self._protein_contiguity_error(topology, companion_pdb)

    def _charmm_validation_error(
        self,
        psf_path: Path,
        pdb_path: Path,
        *,
        expected_glycan: Optional[Mapping[str, Any]],
    ) -> Optional[str]:
        if not self._pdb_contains_expected_glycans(pdb_path, expected_glycan):
            return "Final CHARMM/OpenMM structure is missing the expected glycan residues."

        try:
            from openmm.app import CharmmPsfFile
        except Exception as exc:
            return f"OpenMM CharmmPsfFile import failed while validating CHARMM-GUI CHARMM inputs: {exc}"

        topology = CharmmPsfFile(str(psf_path)).topology
        return self._protein_contiguity_error(topology, pdb_path)

    def _build_charmm_result_from_extracted_tree(
        self,
        *,
        openmm_md,
        model_root: Path,
        extract_root: Path,
        pdb_reader_jobid: str,
        quick_bilayer_jobid: Optional[str] = None,
        membrane_builder_jobid: Optional[str] = None,
        ffconverter_jobid: Optional[str] = None,
        request_payload: Mapping[str, Any],
        validation_note: Optional[str] = None,
    ) -> ParameterizationResult:
        openmm_root = extract_root / "openmm"
        psf_path = _select_best_path(openmm_root.rglob("*.psf"))
        coordinate_path = _select_best_path(openmm_root.rglob("*.crd"), stem_hint="step5_input")
        pdb_path = _select_best_path(openmm_root.rglob("*.pdb"), stem_hint="step5_input")
        toppar_path = openmm_root / "toppar.str"
        sysinfo_path = openmm_root / "sysinfo.dat"
        toppar_dir = extract_root / "toppar"

        missing = [
            str(path_name)
            for path_name, path_value in (
                ("psf", psf_path),
                ("coordinates", coordinate_path),
                ("pdb", pdb_path),
                ("toppar.str", toppar_path if toppar_path.exists() else None),
                ("sysinfo.dat", sysinfo_path if sysinfo_path.exists() else None),
                ("toppar/", toppar_dir if toppar_dir.exists() else None),
            )
            if path_value is None
        ]
        if missing:
            raise FileNotFoundError(
                f"Could not locate the required CHARMM/OpenMM export files in {extract_root}: {', '.join(missing)}."
            )

        selected_manifest = {
            "input_format": "charmm",
            "files": {
                "psf": os.path.relpath(psf_path, model_root),
                "crd": os.path.relpath(coordinate_path, model_root),
                "coordinates": os.path.relpath(coordinate_path, model_root),
                "pdb": os.path.relpath(pdb_path, model_root),
                "toppar_str": os.path.relpath(toppar_path, model_root),
                "sysinfo": os.path.relpath(sysinfo_path, model_root),
                "toppar_dir": os.path.relpath(toppar_dir, model_root),
            },
            "metadata": {
                "backend": "charmm_gui",
                "pdb_reader_jobid": str(pdb_reader_jobid),
                "request": request_payload,
                "extract_root": os.path.relpath(extract_root, model_root),
            },
        }
        if quick_bilayer_jobid is not None:
            selected_manifest["metadata"]["quick_bilayer_jobid"] = str(quick_bilayer_jobid)
        if membrane_builder_jobid is not None:
            selected_manifest["metadata"]["membrane_builder_jobid"] = str(membrane_builder_jobid)
        if ffconverter_jobid is not None:
            selected_manifest["metadata"]["ffconverter_jobid"] = str(ffconverter_jobid)
        if validation_note:
            selected_manifest["metadata"]["validation_note"] = str(validation_note)
        self._write_json(model_root / "selected_inputs.json", selected_manifest)

        result = ParameterizationResult(input_format="charmm")
        for key, relative_path in selected_manifest["files"].items():
            result.with_file(key, str((model_root / relative_path).resolve()))
        result.metadata.update(selected_manifest["metadata"])

        openmm_md.psf_file = str(psf_path)
        openmm_md.crd_file = str(coordinate_path)
        openmm_md.pdb_file = str(pdb_path)
        openmm_md.toppar_str = str(toppar_path)
        openmm_md.sysinfo_file = str(sysinfo_path)
        return result

    def _build_result_from_extracted_tree(
        self,
        *,
        openmm_md,
        model_root: Path,
        extract_root: Path,
        pdb_reader_jobid: str,
        quick_bilayer_jobid: Optional[str] = None,
        membrane_builder_jobid: Optional[str] = None,
        ffconverter_jobid: Optional[str] = None,
        request_payload: Mapping[str, Any],
    ) -> ParameterizationResult:
        expected_glycan = self._extract_expected_glycan_metadata(extract_root)

        topology_candidates = []
        for suffix in _TOPOLOGY_SUFFIXES:
            topology_candidates.extend(extract_root.rglob(f"*{suffix}"))
        topology_path = _select_best_path(topology_candidates)
        coordinate_candidates = []
        for suffix in _COORDINATE_SUFFIXES:
            coordinate_candidates.extend(extract_root.rglob(f"*{suffix}"))

        amber_validation_note = None
        if topology_path is not None:
            coordinate_path = _select_best_path(
                coordinate_candidates,
                directory_hint=topology_path.parent,
                stem_hint=topology_path.stem,
            )
            if coordinate_path is not None:
                amber_validation_note = self._amber_validation_error(
                    topology_path,
                    extract_root=extract_root,
                    expected_glycan=expected_glycan,
                )
                if amber_validation_note is None:
                    selected_manifest = {
                        "input_format": "amber",
                        "files": {
                            "prmtop": os.path.relpath(topology_path, model_root),
                            "coordinates": os.path.relpath(coordinate_path, model_root),
                        },
                        "metadata": {
                            "backend": "charmm_gui",
                            "pdb_reader_jobid": str(pdb_reader_jobid),
                            "request": request_payload,
                            "extract_root": os.path.relpath(extract_root, model_root),
                        },
                    }
                    if quick_bilayer_jobid is not None:
                        selected_manifest["metadata"]["quick_bilayer_jobid"] = str(quick_bilayer_jobid)
                    if membrane_builder_jobid is not None:
                        selected_manifest["metadata"]["membrane_builder_jobid"] = str(membrane_builder_jobid)
                    if ffconverter_jobid is not None:
                        selected_manifest["metadata"]["ffconverter_jobid"] = str(ffconverter_jobid)
                    self._write_json(model_root / "selected_inputs.json", selected_manifest)

                    result = ParameterizationResult(input_format="amber")
                    result.with_file("prmtop", str(topology_path))
                    coordinate_key = coordinate_path.suffix.lstrip(".").lower() or "coordinates"
                    result.with_file(coordinate_key, str(coordinate_path))
                    result.with_file("coordinates", str(coordinate_path))
                    result.metadata.update(selected_manifest["metadata"])

                    openmm_md.prmtop_file = str(topology_path)
                    openmm_md.inpcrd_file = str(coordinate_path)
                    return result

        openmm_psf_path = _select_best_path((extract_root / "openmm").rglob("*.psf"), stem_hint="step5_input")
        openmm_pdb_path = _select_best_path((extract_root / "openmm").rglob("*.pdb"), stem_hint="step5_input")
        charmm_validation_note = None
        if openmm_psf_path is not None and openmm_pdb_path is not None:
            charmm_validation_note = self._charmm_validation_error(
                openmm_psf_path,
                openmm_pdb_path,
                expected_glycan=expected_glycan,
            )
            if charmm_validation_note is None:
                validation_note = None
                if amber_validation_note:
                    validation_note = f"AMBER export was rejected; using CHARMM/OpenMM export instead. {amber_validation_note}"
                return self._build_charmm_result_from_extracted_tree(
                    openmm_md=openmm_md,
                    model_root=model_root,
                    extract_root=extract_root,
                    pdb_reader_jobid=pdb_reader_jobid,
                    quick_bilayer_jobid=quick_bilayer_jobid,
                    membrane_builder_jobid=membrane_builder_jobid,
                    ffconverter_jobid=ffconverter_jobid,
                    request_payload=request_payload,
                    validation_note=validation_note,
                )

        error_messages: list[str] = []
        if topology_path is None:
            error_messages.append(
                f"Could not locate an AMBER topology file ({', '.join(_TOPOLOGY_SUFFIXES)}) in {extract_root}."
            )
        elif amber_validation_note:
            error_messages.append(f"AMBER validation failed: {amber_validation_note}")
        else:
            error_messages.append(
                f"Could not locate an AMBER coordinates file ({', '.join(_COORDINATE_SUFFIXES)}) in {extract_root}."
            )
        if charmm_validation_note:
            error_messages.append(f"CHARMM/OpenMM validation failed: {charmm_validation_note}")
        elif openmm_psf_path is None or openmm_pdb_path is None:
            error_messages.append("Could not locate a complete CHARMM/OpenMM export bundle in the extracted tree.")
        raise RuntimeError(" ".join(error_messages))

    def _build_pending_result(
        self,
        *,
        pdb_reader_jobid: str,
        quick_bilayer_jobid: Optional[str] = None,
        membrane_builder_jobid: Optional[str] = None,
        ffconverter_jobid: Optional[str] = None,
        request_payload: Mapping[str, Any],
        state: str = "submitted",
        module: Optional[str] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> ParameterizationResult:
        result = ParameterizationResult(input_format="remote_pending")
        metadata = {
            "backend": "charmm_gui",
            "state": str(state),
            "pdb_reader_jobid": str(pdb_reader_jobid),
            "request": request_payload,
        }
        if quick_bilayer_jobid is not None:
            metadata["quick_bilayer_jobid"] = str(quick_bilayer_jobid)
        if membrane_builder_jobid is not None:
            metadata["membrane_builder_jobid"] = str(membrane_builder_jobid)
        if ffconverter_jobid is not None:
            metadata["ffconverter_jobid"] = str(ffconverter_jobid)
        if module is None:
            if membrane_builder_jobid is not None:
                module = "membrane_builder"
            elif quick_bilayer_jobid is not None:
                module = "quick_bilayer"
        if module is not None:
            metadata["module"] = str(module)
        if extra_metadata:
            metadata.update(dict(extra_metadata))
        result.metadata.update(metadata)
        return result

    def _extracted_tree_has_amber_inputs(self, extract_root: Path) -> bool:
        topology_candidates: list[Path] = []
        for suffix in _TOPOLOGY_SUFFIXES:
            topology_candidates.extend(extract_root.rglob(f"*{suffix}"))
        topology_path = _select_best_path(topology_candidates)
        if topology_path is None:
            return False

        coordinate_candidates: list[Path] = []
        for suffix in _COORDINATE_SUFFIXES:
            coordinate_candidates.extend(extract_root.rglob(f"*{suffix}"))
        coordinate_path = _select_best_path(
            coordinate_candidates,
            directory_hint=topology_path.parent,
            stem_hint=topology_path.stem,
        )
        return coordinate_path is not None

    def _load_cached_result(self, openmm_md, model_root: Path) -> Optional[ParameterizationResult]:
        manifest_path = model_root / "selected_inputs.json"
        if not manifest_path.exists():
            return None
        manifest = self._read_json(manifest_path)
        files = manifest.get("files", {})
        input_format = str(manifest.get("input_format", "amber")).strip().lower()

        if input_format == "amber":
            topology_rel = files.get("prmtop")
            coordinates_rel = files.get("coordinates")
            if not topology_rel or not coordinates_rel:
                return None

            topology_path = (model_root / topology_rel).resolve()
            coordinates_path = (model_root / coordinates_rel).resolve()
            if not topology_path.exists() or not coordinates_path.exists():
                return None

            extract_root_rel = manifest.get("metadata", {}).get("extract_root")
            extract_root = (model_root / extract_root_rel).resolve() if extract_root_rel else topology_path.parent.parent
            amber_error = self._amber_validation_error(
                topology_path,
                extract_root=extract_root,
                expected_glycan=self._extract_expected_glycan_metadata(extract_root),
            )
            if amber_error is not None:
                return None

            result = ParameterizationResult(input_format="amber")
            result.with_file("prmtop", str(topology_path))
            coordinate_key = coordinates_path.suffix.lstrip(".").lower() or "coordinates"
            result.with_file(coordinate_key, str(coordinates_path))
            result.with_file("coordinates", str(coordinates_path))
            result.metadata.update(dict(manifest.get("metadata", {})))

            openmm_md.prmtop_file = str(topology_path)
            openmm_md.inpcrd_file = str(coordinates_path)
            return result

        if input_format == "charmm":
            required_keys = ("psf", "coordinates", "pdb", "toppar_str", "sysinfo", "toppar_dir")
            resolved_paths = {}
            for key in required_keys:
                relative_path = files.get(key)
                if not relative_path:
                    return None
                resolved_path = (model_root / relative_path).resolve()
                if not resolved_path.exists():
                    return None
                resolved_paths[key] = resolved_path

            charmm_error = self._charmm_validation_error(
                resolved_paths["psf"],
                resolved_paths["pdb"],
                expected_glycan=self._extract_expected_glycan_metadata(
                    (model_root / manifest.get("metadata", {}).get("extract_root", ".")).resolve()
                ),
            )
            if charmm_error is not None:
                return None

            result = ParameterizationResult(input_format="charmm")
            for key, resolved_path in resolved_paths.items():
                result.with_file(key, str(resolved_path))
            result.with_file("crd", str(resolved_paths["coordinates"]))
            result.metadata.update(dict(manifest.get("metadata", {})))

            openmm_md.psf_file = str(resolved_paths["psf"])
            openmm_md.crd_file = str(resolved_paths["coordinates"])
            openmm_md.pdb_file = str(resolved_paths["pdb"])
            openmm_md.toppar_str = str(resolved_paths["toppar_str"])
            openmm_md.sysinfo_file = str(resolved_paths["sysinfo"])
            return result

        return None

    def _load_cached_pdb_reader_jobid(self, model_root: Path) -> Optional[str]:
        manifest_path = model_root / "pdb_reader_manifest.json"
        if manifest_path.exists():
            manifest = self._read_json(manifest_path)
            jobid = str(manifest.get("jobid", "")).strip()
            if jobid:
                return jobid

        request_path = model_root / "request.json"
        if request_path.exists():
            request_payload = self._read_json(request_path)
            query = request_payload.get("query", {})
            if isinstance(query, Mapping):
                jobid = str(query.get("jobid", "")).strip()
                if jobid:
                    return jobid
        return None

    def _load_cached_request_payload(self, model_root: Path) -> Optional[Dict[str, Any]]:
        request_path = model_root / "request.json"
        if not request_path.exists():
            return None
        payload = self._read_json(request_path)
        query = payload.get("query")
        form = payload.get("form")
        if not isinstance(query, Mapping) or not isinstance(form, Mapping):
            return None
        return payload

    def _resolve_membrane_builder_jobid(
        self,
        *,
        model_root: Path,
        options: Mapping[str, Any],
        model_name: str,
        candidate_jobid: str,
        pdb_reader_jobid: str,
        submitted_now: bool,
        site_base_url: str,
        submission_epoch: int,
        retriever_snapshot_path: Path,
        jobids_snapshot_path: Path,
    ) -> str:
        cached_jobid = self._load_cached_membrane_builder_jobid(model_root)
        if cached_jobid is not None:
            return cached_jobid

        email, password = self._resolve_login_credentials(options, model_name)
        if not email or not password:
            return candidate_jobid or pdb_reader_jobid

        session = self._create_site_session(site_base_url, email, password)
        activation_timeout_s = float(options.get("membrane_builder_activation_timeout_s", 120.0))
        activation_poll_interval_s = float(options.get("membrane_builder_activation_poll_interval_s", 5.0))
        if not submitted_now:
            activation_timeout_s = min(activation_timeout_s, 5.0)
            activation_poll_interval_s = min(activation_poll_interval_s, 1.0)

        activation = self._wait_for_membrane_builder_activation(
            session=session,
            site_base_url=site_base_url,
            jobid=candidate_jobid,
            pdb_reader_jobid=pdb_reader_jobid,
            retriever_snapshot_path=retriever_snapshot_path,
            jobids_snapshot_path=jobids_snapshot_path,
            poll_interval_s=activation_poll_interval_s,
            timeout_s=activation_timeout_s,
            submission_epoch=submission_epoch,
        )
        self._write_json(model_root / "membrane_builder_manifest.json", activation)
        return str(activation.get("jobid", candidate_jobid)).strip() or candidate_jobid or pdb_reader_jobid

    def _wait_for_membrane_builder_activation(
        self,
        *,
        session,
        site_base_url: str,
        jobid: str,
        pdb_reader_jobid: str,
        retriever_snapshot_path: Path,
        jobids_snapshot_path: Path,
        poll_interval_s: float,
        timeout_s: float,
        submission_epoch: int,
    ) -> Dict[str, Any]:
        start_time = time.monotonic()
        last_steps: list[str] = []
        lookup_jobids = [str(jobid).strip()]
        parent_jobid = str(pdb_reader_jobid).strip()
        if parent_jobid and parent_jobid not in lookup_jobids:
            lookup_jobids.append(parent_jobid)

        while True:
            last_steps = []
            for lookup_jobid in lookup_jobids:
                html_text = self._site_request(
                    session,
                    f"{site_base_url}/?doc=input/retriever&jobid={lookup_jobid}",
                    method="GET",
                ).decode("utf-8", errors="replace")
                self._write_text(retriever_snapshot_path, html_text)

                builder_step = self._extract_membrane_builder_step_from_retriever(html_text)
                if builder_step is not None:
                    builder_step["source_jobid"] = str(lookup_jobid)
                    return builder_step

                parsed_steps = self._parse_retriever_steps(html_text)
                if parsed_steps:
                    last_steps = [f"{step['doc']}:{step['step']}" for step in parsed_steps]

            jobids_html = self._site_request(
                session,
                f"{site_base_url}/?doc=user/jobids",
                method="GET",
            ).decode("utf-8", errors="replace")
            self._write_text(jobids_snapshot_path, jobids_html)
            builder_step = self._extract_membrane_builder_step_from_jobids_page(
                jobids_html,
                candidate_jobid=jobid,
                pdb_reader_jobid=pdb_reader_jobid,
                submission_epoch=submission_epoch,
            )
            if builder_step is not None:
                builder_step["source_jobid"] = "jobids"
                return builder_step

            if time.monotonic() - start_time >= timeout_s:
                recovered = ", ".join(last_steps) if last_steps else "none"
                raise RuntimeError(
                    f"CHARMM-GUI Membrane Builder submission for job {jobid} did not activate. "
                    f"The retriever page still exposes only: {recovered}."
                )
            time.sleep(poll_interval_s)

    def _resolve_quick_bilayer_jobid(
        self,
        *,
        model_root: Path,
        options: Mapping[str, Any],
        model_name: str,
        candidate_jobid: str,
        pdb_reader_jobid: str,
        submitted_now: bool,
        workflow_mode: str,
        site_base_url: str,
        submission_epoch: int,
        quick_page_snapshot_path: Path,
        jobids_snapshot_path: Path,
    ) -> str:
        cached_jobid = self._load_cached_quick_bilayer_jobid(model_root)
        if cached_jobid is not None:
            return cached_jobid

        verify_submission = _normalize_bool(options.get("verify_quick_bilayer_submission"), default=True)
        if not verify_submission:
            return candidate_jobid

        email, password = self._resolve_login_credentials(options, model_name)
        if not email or not password:
            return candidate_jobid

        session = self._create_site_session(site_base_url, email, password)
        activation_timeout_s = float(options.get("quick_bilayer_activation_timeout_s", 120.0))
        activation_poll_interval_s = float(options.get("quick_bilayer_activation_poll_interval_s", 5.0))
        if not submitted_now:
            activation_timeout_s = min(activation_timeout_s, 5.0)
            activation_poll_interval_s = min(activation_poll_interval_s, 1.0)

        activation = self._wait_for_quick_bilayer_activation(
            session=session,
            site_base_url=site_base_url,
            jobid=candidate_jobid,
            pdb_reader_jobid=pdb_reader_jobid,
            retriever_snapshot_path=model_root / "quick_bilayer_retriever.html",
            quick_page_snapshot_path=quick_page_snapshot_path,
            jobids_snapshot_path=jobids_snapshot_path,
            poll_interval_s=activation_poll_interval_s,
            timeout_s=activation_timeout_s,
            workflow_mode=workflow_mode,
            submission_epoch=submission_epoch,
        )
        self._write_json(model_root / "quick_bilayer_manifest.json", activation)
        return str(activation.get("jobid", candidate_jobid)).strip() or candidate_jobid

    def _wait_for_quick_bilayer_activation(
        self,
        *,
        session,
        site_base_url: str,
        jobid: str,
        pdb_reader_jobid: str,
        retriever_snapshot_path: Path,
        quick_page_snapshot_path: Path,
        jobids_snapshot_path: Path,
        poll_interval_s: float,
        timeout_s: float,
        workflow_mode: str,
        submission_epoch: int,
    ) -> Dict[str, Any]:
        start_time = time.monotonic()
        last_steps: list[str] = []
        lookup_jobids = [str(jobid).strip()]
        parent_jobid = str(pdb_reader_jobid).strip()
        if parent_jobid and parent_jobid not in lookup_jobids:
            lookup_jobids.append(parent_jobid)
        while True:
            last_steps = []
            for lookup_jobid in lookup_jobids:
                html_text = self._site_request(
                    session,
                    f"{site_base_url}/?doc=input/retriever&jobid={lookup_jobid}",
                    method="GET",
                ).decode("utf-8", errors="replace")
                self._write_text(retriever_snapshot_path, html_text)

                quick_step = self._extract_quick_bilayer_step_from_retriever(html_text)
                if quick_step is not None:
                    quick_step["source_jobid"] = str(lookup_jobid)
                    return quick_step

                quick_page_html = self._site_request(
                    session,
                    f"{site_base_url}/?doc=input/membrane.quick&jobid={lookup_jobid}&project=pdbreader",
                    method="GET",
                ).decode("utf-8", errors="replace")
                self._write_text(quick_page_snapshot_path, quick_page_html)
                quick_step = self._extract_quick_bilayer_step_from_quick_page(quick_page_html, lookup_jobid)
                if quick_step is not None:
                    quick_step["source_jobid"] = str(lookup_jobid)
                    return quick_step

                parsed_steps = self._parse_retriever_steps(html_text)
                if parsed_steps:
                    last_steps = [f"{step['doc']}:{step['step']}" for step in parsed_steps]

            jobids_html = self._site_request(
                session,
                f"{site_base_url}/?doc=user/jobids",
                method="GET",
            ).decode("utf-8", errors="replace")
            self._write_text(jobids_snapshot_path, jobids_html)
            quick_step = self._extract_quick_bilayer_step_from_jobids_page(
                jobids_html,
                candidate_jobid=jobid,
                pdb_reader_jobid=pdb_reader_jobid,
                submission_epoch=submission_epoch,
            )
            if quick_step is not None:
                quick_step["source_jobid"] = "jobids"
                return quick_step
            if time.monotonic() - start_time >= timeout_s:
                recovered = ", ".join(last_steps) if last_steps else "none"
                raise RuntimeError(
                    f"CHARMM-GUI Quick Bilayer submission for job {jobid} did not activate. "
                    f"The retriever page still exposes only: {recovered}. "
                    f"Expected a membrane.quick step before continuing in workflow_mode={workflow_mode!r}."
                )
            time.sleep(poll_interval_s)

    def _extract_quick_bilayer_step_from_retriever(self, html_text: str) -> Optional[Dict[str, Any]]:
        for step in self._parse_retriever_steps(html_text):
            doc_name = str(step.get("doc", "")).strip().lower()
            project_name = str(step.get("project", "")).strip().lower()
            if doc_name == "membrane.quick" or project_name == "membrane_quick":
                return step
        return None

    def _extract_membrane_builder_step_from_retriever(self, html_text: str) -> Optional[Dict[str, Any]]:
        best_step: Optional[Dict[str, Any]] = None
        best_rank = -1
        for step in self._parse_retriever_steps(html_text):
            doc_name = str(step.get("doc", "")).strip().lower()
            if doc_name == "membrane.bilayer":
                try:
                    step_rank = int(str(step.get("step", "")).strip())
                except ValueError:
                    step_rank = -1
                if best_step is None or step_rank > best_rank:
                    best_step = step
                    best_rank = step_rank
        return best_step

    def _extract_ffconverter_step_from_retriever(
        self,
        html_text: str,
        *,
        expected_project: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        best_step: Optional[Dict[str, Any]] = None
        best_rank = -1
        normalized_expected_project = str(expected_project).strip().lower() if expected_project else ""
        for step in self._parse_retriever_steps(html_text):
            doc_name = str(step.get("doc", "")).strip().lower()
            if doc_name != "converter.ffconverter":
                continue
            project_name = str(step.get("project", "")).strip().lower()
            if normalized_expected_project and project_name != normalized_expected_project:
                continue
            try:
                step_rank = int(str(step.get("step", "")).strip())
            except ValueError:
                step_rank = -1
            if best_step is None or step_rank > best_rank:
                best_step = step
                best_rank = step_rank
        return best_step

    def _get_live_retriever_status(
        self,
        *,
        site_base_url: str,
        email: str,
        password: str,
        jobid: str,
        retriever_snapshot_path: Path,
        ffconverter_project: Optional[str] = "pdbreader",
    ) -> Dict[str, Any]:
        session = self._create_site_session(site_base_url, email, password)
        html_text = self._site_request(
            session,
            f"{site_base_url}/?doc=input/retriever&jobid={jobid}",
            method="GET",
        ).decode("utf-8", errors="replace")
        self._write_text(retriever_snapshot_path, html_text)
        steps = self._parse_retriever_steps(html_text)
        return {
            "steps": [f"{step['doc']}:{step['step']}" for step in steps],
            "quick_bilayer_step": self._extract_quick_bilayer_step_from_retriever(html_text),
            "ffconverter_step": self._extract_ffconverter_step_from_retriever(
                html_text,
                expected_project=ffconverter_project,
            ),
        }

    def _extract_quick_bilayer_step_from_quick_page(self, html_text: str, jobid: str) -> Optional[Dict[str, Any]]:
        doc_match = re.search(r"""var\s+doc\s*=\s*['"](?P<doc>[^'"]+)['"]""", html_text, re.IGNORECASE)
        step_match = re.search(r"""var\s+step\s*=\s*['"](?P<step>[^'"]+)['"]""", html_text, re.IGNORECASE)
        project_match = re.search(r"""var\s+project\s*=\s*['"](?P<project>[^'"]+)['"]""", html_text, re.IGNORECASE)
        if not doc_match or not step_match:
            return None
        doc_name = doc_match.group("doc").strip().lower()
        if not doc_name.endswith("membrane.quick"):
            return None
        step_number = step_match.group("step").strip()
        if not step_number or not step_number.isdigit():
            return None
        project_name = project_match.group("project").strip() if project_match else "pdbreader"
        return {
            "doc": "membrane.quick",
            "step": step_number,
            "project": project_name or "pdbreader",
            "jobid": str(jobid).strip(),
        }

    def _parse_retriever_steps(self, html_text: str) -> list[Dict[str, Any]]:
        steps: list[Dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for match in _RETRIEVER_LINK_PATTERN.finditer(html_text):
            doc_name = html.unescape(match.group("doc")).strip()
            step_number = html.unescape(match.group("step")).strip()
            project_name = html.unescape(match.group("project")).strip()
            jobid = html.unescape(match.group("jobid")).strip()
            key = (doc_name, step_number, project_name, jobid)
            if key in seen:
                continue
            seen.add(key)
            steps.append(
                {
                    "doc": doc_name,
                    "step": step_number,
                    "project": project_name,
                    "jobid": jobid,
                }
            )
        return steps

    def _extract_quick_bilayer_step_from_jobids_page(
        self,
        html_text: str,
        *,
        candidate_jobid: str,
        pdb_reader_jobid: str,
        submission_epoch: int,
    ) -> Optional[Dict[str, Any]]:
        rows = self._parse_jobids_rows(html_text)
        exact_jobids = {str(candidate_jobid).strip(), str(pdb_reader_jobid).strip()}
        for row in rows:
            module_name = str(row.get("module", "")).strip().lower()
            if module_name != "quick bilayer":
                continue
            if str(row.get("jobid", "")).strip() not in exact_jobids:
                continue
            return {
                "doc": "membrane.quick",
                "step": str(row.get("step", "")).strip(),
                "project": "pdbreader",
                "jobid": str(row.get("jobid", "")).strip(),
                "module": str(row.get("module", "")).strip(),
                "status": str(row.get("status", "")).strip(),
            }

        recent_quick_rows = []
        for row in rows:
            module_name = str(row.get("module", "")).strip().lower()
            if module_name != "quick bilayer":
                continue
            created_epoch = row.get("created_epoch")
            if not isinstance(created_epoch, int):
                continue
            if created_epoch < submission_epoch - 10:
                continue
            recent_quick_rows.append(row)

        if len(recent_quick_rows) == 1:
            row = recent_quick_rows[0]
            return {
                "doc": "membrane.quick",
                "step": str(row.get("step", "")).strip(),
                "project": "pdbreader",
                "jobid": str(row.get("jobid", "")).strip(),
                "module": str(row.get("module", "")).strip(),
                "status": str(row.get("status", "")).strip(),
            }

        return None

    def _extract_membrane_builder_step_from_jobids_page(
        self,
        html_text: str,
        *,
        candidate_jobid: str,
        pdb_reader_jobid: str,
        submission_epoch: int,
    ) -> Optional[Dict[str, Any]]:
        rows = self._parse_jobids_rows(html_text)
        exact_jobids = {str(candidate_jobid).strip(), str(pdb_reader_jobid).strip()}
        for row in rows:
            module_name = str(row.get("module", "")).strip().lower()
            if module_name != "bilayer builder":
                continue
            if str(row.get("jobid", "")).strip() not in exact_jobids:
                continue
            return {
                "doc": "membrane.bilayer",
                "step": str(row.get("step", "")).strip(),
                "project": "pdbreader",
                "jobid": str(row.get("jobid", "")).strip(),
                "module": str(row.get("module", "")).strip(),
                "status": str(row.get("status", "")).strip(),
            }

        recent_builder_rows = []
        for row in rows:
            module_name = str(row.get("module", "")).strip().lower()
            if module_name != "bilayer builder":
                continue
            created_epoch = row.get("created_epoch")
            if not isinstance(created_epoch, int):
                continue
            if created_epoch < submission_epoch - 10:
                continue
            recent_builder_rows.append(row)

        if len(recent_builder_rows) == 1:
            row = recent_builder_rows[0]
            return {
                "doc": "membrane.bilayer",
                "step": str(row.get("step", "")).strip(),
                "project": "pdbreader",
                "jobid": str(row.get("jobid", "")).strip(),
                "module": str(row.get("module", "")).strip(),
                "status": str(row.get("status", "")).strip(),
            }

        return None

    def _get_jobids_page_status(
        self,
        session,
        *,
        site_base_url: str,
        jobid: str,
        jobids_snapshot_path: Path,
        job_modules: Optional[Sequence[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        jobids_html = self._site_request(
            session,
            f"{site_base_url}/?doc=user/jobids",
            method="GET",
        ).decode("utf-8", errors="replace")
        self._write_text(jobids_snapshot_path, jobids_html)
        normalized_modules = {
            str(module_name).strip().lower() for module_name in (job_modules or ("Quick Bilayer",))
        }
        for row in self._parse_jobids_rows(jobids_html):
            module_name = str(row.get("module", "")).strip().lower()
            if module_name not in normalized_modules:
                continue
            if str(row.get("jobid", "")).strip() != str(jobid).strip():
                continue
            return {
                "status": str(row.get("status", "")).strip(),
                "jobid": str(row.get("jobid", "")).strip(),
                "step": str(row.get("step", "")).strip(),
                "source": "jobids_page",
                "module": str(row.get("module", "")).strip(),
                "project": str(row.get("project", "")).strip(),
                "control": str(row.get("control", "")).strip(),
                "created_epoch": row.get("created_epoch"),
                "hasTarFile": str(row.get("status", "")).strip().lower() in _DONE_STATUSES,
            }
        return None

    def _parse_jobids_rows(self, html_text: str) -> list[Dict[str, Any]]:
        parser = _HTMLTableParser()
        parser.feed(html_text)
        rows: list[Dict[str, Any]] = []
        for cells in parser.rows:
            if len(cells) < 7:
                continue
            if cells[0]["text"] == "Job ID":
                continue
            jobid = str(cells[0]["text"]).strip()
            if not jobid.isdigit():
                continue
            created_text = str(cells[6]["attrs"].get("data-value") or "").strip()
            created_epoch = int(created_text) if created_text.isdigit() else None
            rows.append(
                {
                    "jobid": jobid,
                    "project": str(cells[1]["text"]).strip(),
                    "module": str(cells[2]["text"]).strip(),
                    "step": str(cells[3]["text"]).strip(),
                    "status": str(cells[4]["text"]).strip(),
                    "control": str(cells[5]["text"]).strip(),
                    "created_epoch": created_epoch,
                }
            )
        return rows

    def _load_cached_quick_bilayer_jobid(self, model_root: Path) -> Optional[str]:
        manifest_path = model_root / "quick_bilayer_manifest.json"
        if not manifest_path.exists():
            return None
        manifest = self._read_json(manifest_path)
        jobid = str(manifest.get("jobid", "")).strip()
        if not jobid:
            return None
        return jobid

    def _load_cached_membrane_builder_jobid(self, model_root: Path) -> Optional[str]:
        manifest_path = model_root / "membrane_builder_manifest.json"
        if not manifest_path.exists():
            return None
        manifest = self._read_json(manifest_path)
        jobid = str(manifest.get("jobid", "")).strip()
        if not jobid:
            return None
        return jobid

    def _load_cached_membrane_builder_step(self, model_root: Path) -> Optional[str]:
        manifest_path = model_root / "membrane_builder_manifest.json"
        if not manifest_path.exists():
            return None
        manifest = self._read_json(manifest_path)
        step = str(manifest.get("step", "")).strip()
        if not step:
            return None
        return step

    def _http_json(
        self,
        url: str,
        token: Optional[str],
        *,
        method: str,
        data: Optional[bytes] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        payload = self._http_request(url, token, method=method, data=data, headers=headers)
        try:
            return json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"CHARMM-GUI API returned non-JSON data for {url!r}.") from exc

    def _http_bytes(
        self,
        url: str,
        token: Optional[str],
        *,
        method: str,
        data: Optional[bytes] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> bytes:
        return self._http_request(url, token, method=method, data=data, headers=headers)

    def _http_request(
        self,
        url: str,
        token: Optional[str],
        *,
        method: str,
        data: Optional[bytes] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> bytes:
        request_headers = {
            "Accept": "application/json",
        }
        if token:
            request_headers["Authorization"] = f"Bearer {token}"
        if headers:
            request_headers.update({str(key): str(value) for key, value in headers.items()})
        request = urllib_request.Request(url, data=data, headers=request_headers, method=method)
        timeout_s = float(self.options.get("api_request_timeout_s", 120.0))
        try:
            try:
                response = _urlopen(request, timeout=timeout_s)
            except TypeError:
                response = _urlopen(request)
            with response:
                return response.read()
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"CHARMM-GUI API request failed with HTTP {exc.code}: {body}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Failed to reach CHARMM-GUI API at {url}: {exc}") from exc

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    @staticmethod
    def _write_text(path: Path, contents: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(contents)

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {path}, found {type(payload).__name__}.")
        return payload
