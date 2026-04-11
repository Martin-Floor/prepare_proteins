import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from prepare_proteins.MD.parameterization.base import ParameterizationResult
from prepare_proteins.MD.parameterization.charmm_gui import CHARMMGUIBackend
import prepare_proteins.MD.parameterization.charmm_gui as charmm_gui_module


class _FakeHTTPResponse(io.BytesIO):
    def __init__(self, payload: bytes, *, headers: dict[str, str] | None = None):
        super().__init__(payload)
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class _FakeSiteOpener:
    def __init__(self, handler):
        self.handler = handler

    def open(self, request):
        return self.handler(request)


def _make_tarball(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, contents in entries.items():
            payload = contents.encode("utf-8")
            member = tarfile.TarInfo(name=name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return buffer.getvalue()


def _build_openmm_md(tmp_path):
    input_pdb = tmp_path / "modelA.pdb"
    input_pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n")
    return SimpleNamespace(
        pdb_name="modelA",
        input_pdb=str(input_pdb),
        command_log=[],
    )


def _make_pdb_reader_chain_selection_html(jobid="12345"):
    return f"""
    <html>
      <body>
        <span class='jobid'>JOB ID: {jobid}</span>
        <form id='fpdbreader' name='fpdbreader' method='post' action='./?doc=input/pdbreader&step=2'>
          <input type='hidden' name='jobid' value='{jobid}'>
          <input type='hidden' name='pdb_id' value='modelA'>
          <input type='hidden' name='source' value='RCSB'>
          <input type='hidden' name='pdb_format' value='PDB'>
          <input type='hidden' name='project' value='pdbreader'>
          <input type='hidden' name='model_id' value='0'>
          <input type='checkbox' name='chains[PROA][checked]' value='1' checked>
          <input type='text' name='chains[PROA][segid]' value='PROA'>
          <input type='text' name='chains[PROA][first]' value='1'>
          <input type='text' name='chains[PROA][last]' value='10'>
          <input type='checkbox' name='chains[HETO][checked]' value='1' checked>
          <input type='text' name='chains[HETO][segid]' value='HETO'>
        </form>
      </body>
    </html>
    """


def _make_pdb_reader_options_html(jobid="12345"):
    return f"""
    <html>
      <body>
        <form id='fpdbreader' name='fpdbreader' method='post' action='./?doc=input/pdbreader&step=3'>
          <input type='hidden' name='jobid' value='{jobid}'>
          <input type='hidden' name='project' value='pdbreader'>
          <input type='hidden' name='pdb_id' value='modelA'>
          <input type='hidden' name='source' value='RCSB'>
          <input type='hidden' name='pdb_format' value='PDB'>
          <input type='hidden' name='model_id' value='0'>
          <input type='hidden' name='chains[PROA][checked]' value='1'>
          <input type='hidden' name='chains[PROA][segid]' value='PROA'>
          <input type='text' name='system_pH' value='7.0'>
          <input type='checkbox' name='rename_checked' value='1' checked>
          <select name='terminal[PROA][first]'>
            <option value='NTER' selected='selected'>NTER</option>
          </select>
        </form>
      </body>
    </html>
    """


def _make_pdb_reader_ready_html(jobid="12345"):
    return f"""
    <html>
      <body>
        <span class='jobid'>JOB ID: {jobid}</span>
        <script>
          var step = 'reader';
        </script>
      </body>
    </html>
    """


def _make_retriever_html(
    jobid="12345",
    *,
    include_quick_bilayer=False,
    quick_bilayer_jobid=None,
    quick_bilayer_step="2",
    include_membrane_builder=False,
    membrane_builder_jobid=None,
    membrane_builder_step="3",
    membrane_builder_project="pdbreader",
    include_ffconverter=False,
    ffconverter_jobid=None,
    ffconverter_step="6",
    ffconverter_project="pdbreader",
):
    quick_bilayer_jobid = quick_bilayer_jobid or jobid
    quick_row = ""
    if include_quick_bilayer:
        quick_row = f"""
        <tr>
          <td>membrane.quick</td>
          <td>{quick_bilayer_step}</td>
          <td><a href="?doc=input/membrane.quick&step={quick_bilayer_step}&project=membrane_quick&jobid={quick_bilayer_jobid}">Go</a></td>
        </tr>
        """
    membrane_builder_jobid = membrane_builder_jobid or jobid
    membrane_builder_row = ""
    if include_membrane_builder:
        membrane_builder_row = f"""
        <tr>
          <td>membrane.bilayer</td>
          <td>{membrane_builder_step}</td>
          <td><a href="?doc=input/membrane.bilayer&step={membrane_builder_step}&project={membrane_builder_project}&jobid={membrane_builder_jobid}">Go</a></td>
        </tr>
        """
    ffconverter_jobid = ffconverter_jobid or quick_bilayer_jobid
    ffconverter_row = ""
    if include_ffconverter:
        ffconverter_row = f"""
        <tr>
          <td>converter.ffconverter</td>
          <td>{ffconverter_step}</td>
          <td><a href="?doc=input/converter.ffconverter&step={ffconverter_step}&project={ffconverter_project}&jobid={ffconverter_jobid}">Go</a></td>
        </tr>
        """
    return f"""
    <html>
      <body>
        <table id="recovery_table">
          <tr>
            <td>pdbreader</td>
            <td>1</td>
            <td><a href="?doc=input/pdbreader&step=1&project=pdbreader&jobid={jobid}">Go</a></td>
          </tr>
          <tr>
            <td>pdbreader</td>
            <td>2</td>
            <td><a href="?doc=input/pdbreader&step=2&project=pdbreader&jobid={jobid}">Go</a></td>
          </tr>
          <tr>
            <td>pdbreader</td>
            <td>3</td>
            <td><a href="?doc=input/pdbreader&step=3&project=pdbreader&jobid={jobid}">Go</a></td>
          </tr>
          {membrane_builder_row}
          {quick_row}
          {ffconverter_row}
        </table>
      </body>
    </html>
    """


def _make_quick_page_html(jobid="12345", *, step="3", project="pdbreader", doc="input/membrane.quick"):
    return f"""
    <html>
      <body>
        <script>
          var jobid = '{jobid}';
          var project = '{project}';
          var step = '{step}';
          var doc = '{doc}';
        </script>
        <a href="./?doc=input/download&jobid={jobid}" class="download">download.tgz</a>
      </body>
    </html>
    """


def _make_quick_site_submit_html(jobid="12345", *, step="3", project="pdbreader"):
    return f"""
    <html>
      <body>
        <span class='jobid'>JOB ID: {jobid}</span>
        <script>
          var doc = 'input/membrane.quick';
          var step = '{step}';
          var project = '{project}';
        </script>
      </body>
    </html>
    """


def _make_membrane_builder_step1_html(jobid="12345"):
    return f"""
    <html>
      <body>
        <span class='jobid'>JOB ID: {jobid}</span>
        <form id='fpdbreader' name='fpdbreader' method='post' action='./?doc=input/membrane.bilayer&step=3'>
          <input type='hidden' name='jobid' value='{jobid}'>
          <input type='hidden' name='project' value='pdbreader'>
          <input type='hidden' name='lipid_option' value='hetero'>
          <input type='hidden' name='hetero_xy_option' value='ratio'>
          <input type='text' name='hetero_lx' value=''>
          <input type='hidden' name='hetero_z_option' value='wdist'>
          <input type='text' name='hetero_wdist' value='22.5'>
          <input type='checkbox' name='check_penetration' value='1' checked>
          <input type='text' name='lipid_ratio[upper][popc]' value='0'>
          <input type='text' name='lipid_ratio[lower][popc]' value='0'>
          <input type='text' name='lipid_ratio[area][popc]' value='68.3'>
          <input type='text' name='lipid_ratio[name][popc]' value='POPC'>
          <input type='text' name='lipid_ratio[upper][pope]' value='0'>
          <input type='text' name='lipid_ratio[lower][pope]' value='0'>
          <input type='text' name='lipid_ratio[area][pope]' value='58.8'>
          <input type='text' name='lipid_ratio[name][pope]' value='POPE'>
        </form>
      </body>
    </html>
    """


def _make_jobids_html(rows):
    body = []
    for row in rows:
        created_epoch = row.get("created_epoch", 1774815934)
        body.append(
            f"""
            <tr>
              <td><a href="?doc=input/retriever&jobid={row['jobid']}">{row['jobid']}</a></td>
              <td>{row.get('project', 'PDB Reader & Manipulator')}</td>
              <td>{row.get('module', 'Quick Bilayer')}</td>
              <td>{row.get('step', '2')}</td>
              <td>{row.get('status', 'Running')}</td>
              <td>{row.get('control', 'N/A')}</td>
              <td class="timestamp" data-value="{created_epoch}">??</td>
            </tr>
            """
        )
    return f"""
    <html>
      <body>
        <table>
          <tr>
            <td>Job ID</td>
            <td>Project</td>
            <td>Module</td>
            <td>Step</td>
            <td>Status</td>
            <td>Control</td>
            <td>Date Created</td>
          </tr>
          {''.join(body)}
        </table>
      </body>
    </html>
    """


def test_charmm_gui_backend_adds_cached_glycan_fields_to_pdb_reader_submission(tmp_path):
    backend = CHARMMGUIBackend(token="token")
    model_root = tmp_path / "modelA"
    glycan_root = model_root / "extracted" / "charmm-gui-12345"
    glycan_root.mkdir(parents=True)
    (glycan_root / "glycan.yml").write_text(
        "- chain: B\n"
        "  segid: CARA\n"
        "  type: n-linked\n"
        "  grs: |\n"
        "    1 BGLCNA\n"
        "    2 - 14B: BGLCNA\n"
        "  linked_chain: B\n"
        "  linked_resid: 4\n",
        encoding="utf-8",
    )

    fields = [
        ("jobid", "12345"),
        ("rename[NAG]", "NAG"),
        ("newname[NAG]", ""),
    ]
    html_text = """
    <html>
      <body>
        <script>
          var prot_chains = {"PROB": {"4": "ASN"}};
        </script>
      </body>
    </html>
    """

    updated_fields = backend._apply_pdb_reader_manipulation_overrides(
        fields,
        {"include_hetero": True},
        html_text=html_text,
        model_root=model_root,
    )
    updated_map = {name: value for name, value in updated_fields}

    assert updated_map["glyc_checked"] == "1"
    assert updated_map["glycan[CARA][chain]"] == "none"
    assert updated_map["glycan[CARA][type]"] == "n-linked"
    assert updated_map["glycan[CARA][grs]"] == "1 BGLCNA\n2 - 14B: BGLCNA"
    assert json.loads(updated_map["glycan[CARA][prot]"]) == {
        "segid": "PROB",
        "resname": "ASN",
        "resid": "4",
    }
    assert "rename[NAG]" not in updated_map
    assert "newname[NAG]" not in updated_map


def test_charmm_gui_backend_adds_cached_glycan_fields_without_prot_chains(tmp_path):
    backend = CHARMMGUIBackend(token="token")
    model_root = tmp_path / "modelA"
    model_root.mkdir(parents=True)
    (model_root / "modelA.input.pdb").write_text(
        "ATOM      1  N   ASN B   4       0.000   0.000   0.000  1.00 20.00           N\n"
        "ATOM      2  CA  ASN B   4       1.000   0.000   0.000  1.00 20.00           C\n"
        "HETATM    3  C1  NAG G   1       2.000   0.000   0.000  1.00 20.00           C\n"
        "END\n",
        encoding="utf-8",
    )
    (model_root / "glycan.yml").write_text(
        "- chain: B\n"
        "  glycan_list:\n"
        "    B: [1, 2]\n"
        "  grs: |\n"
        "    1 BGLCNA\n"
        "    2 - 14B: BGLCNA\n"
        "  residues:\n"
        "    1: {linked_chain: B, linked_resid: 4, resid: 1, resname: BGLCNA}\n"
        "    2: {linked_chain: B, linked_resid: 1, resid: 2, resname: BGLCNA}\n"
        "  segid: CARA\n"
        "  type: n-linked\n",
        encoding="utf-8",
    )

    fields = [
        ("jobid", "12345"),
        ("rename[NAG]", "rename"),
        ("newname[NAG]", ""),
    ]
    html_text = "<html><body></body></html>"

    updated_fields = backend._apply_pdb_reader_manipulation_overrides(
        fields,
        {"include_hetero": True},
        html_text=html_text,
        model_root=model_root,
    )
    updated_map = {name: value for name, value in updated_fields}

    assert updated_map["glyc_checked"] == "1"
    assert updated_map["glycan[CARA][chain]"] == "none"
    assert updated_map["glycan[CARA][type]"] == "n-linked"
    assert updated_map["glycan[CARA][grs]"] == "1 BGLCNA\n2 - 14B: BGLCNA"
    assert json.loads(updated_map["glycan[CARA][prot]"]) == {
        "segid": "PROB",
        "resname": "ASN",
        "resid": "4",
    }


def test_sanitize_pdb_reader_upload_file_strips_uploaded_glycans(tmp_path):
    model_root = tmp_path / "modelA"
    model_root.mkdir()
    upload_pdb = model_root / "modelA.input.pdb"
    upload_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA B   1       0.000   0.000   0.000  1.00 20.00           C  ",
                "ATOM      2  CA  ASN B   4       1.000   0.000   0.000  1.00 20.00           C  ",
                "ATOM      3  CA  ASP B  64       2.000   0.000   0.000  1.00 20.00           C  ",
                "TER       4      ASP B  64                                                      ",
                "HETATM    5  C1  NAG G   1       3.000   0.000   0.000  1.00 20.00           C  ",
                "HETATM    6  C1  NAG G   2       4.000   0.000   0.000  1.00 20.00           C  ",
                "TER       7      NAG G   2                                                      ",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (model_root / "glycan.yml").write_text(
        "- chain: CARA\n"
        "  segid: B\n"
        "  glycan_list:\n"
        "    linked_chain: B,\n"
        "    linked_resid: 4\n"
        "    residue_list: [1, 2]\n"
        "  type: n-linked\n"
        "  grs: |\n"
        "    1 BGLCNA\n"
        "    2 - 14B: BGLCNA\n",
        encoding="utf-8",
    )

    sanitized = charmm_gui_module._sanitize_pdb_reader_upload_file(upload_pdb, model_root)

    assert sanitized != upload_pdb
    sanitized_text = sanitized.read_text(encoding="utf-8")
    assert "NAG G   1" not in sanitized_text
    assert "NAG G   2" not in sanitized_text
    assert "ALA B   1" in sanitized_text
    assert "ASN B   4" in sanitized_text
    assert "ASP B  64" in sanitized_text


def test_charmm_gui_backend_falls_back_to_charmm_export_when_amber_is_invalid(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend(token="token")
    model_root = tmp_path / "modelA"
    extract_root = model_root / "extracted" / "charmm-gui-12345"
    (extract_root / "amber").mkdir(parents=True)
    (extract_root / "openmm").mkdir(parents=True)
    (extract_root / "toppar").mkdir(parents=True)

    (extract_root / "amber" / "step5_input.parm7").write_text("", encoding="utf-8")
    (extract_root / "amber" / "step5_input.rst7").write_text("", encoding="utf-8")
    (extract_root / "amber" / "step5_input.pdb").write_text("END\n", encoding="utf-8")
    (extract_root / "openmm" / "step5_input.psf").write_text("", encoding="utf-8")
    (extract_root / "openmm" / "step5_input.crd").write_text("", encoding="utf-8")
    (extract_root / "openmm" / "step5_input.pdb").write_text("END\n", encoding="utf-8")
    (extract_root / "openmm" / "toppar.str").write_text("../toppar/dummy.prm\n", encoding="utf-8")
    (extract_root / "openmm" / "sysinfo.dat").write_text('{"dimensions": [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]}\n', encoding="utf-8")
    (extract_root / "toppar" / "dummy.prm").write_text("", encoding="utf-8")

    monkeypatch.setattr(backend, "_extract_expected_glycan_metadata", lambda root: None)
    monkeypatch.setattr(backend, "_amber_validation_error", lambda *args, **kwargs: "broken AMBER export")
    monkeypatch.setattr(backend, "_charmm_validation_error", lambda *args, **kwargs: None)

    openmm_md = SimpleNamespace()
    result = backend._build_result_from_extracted_tree(
        openmm_md=openmm_md,
        model_root=model_root,
        extract_root=extract_root,
        pdb_reader_jobid="12345",
        quick_bilayer_jobid="67890",
        request_payload={"module": "quick_bilayer"},
    )

    manifest = json.loads((model_root / "selected_inputs.json").read_text(encoding="utf-8"))
    assert result.input_format == "charmm"
    assert manifest["input_format"] == "charmm"
    assert "AMBER export was rejected" in manifest["metadata"]["validation_note"]
    assert Path(result.files["psf"]).name == "step5_input.psf"
    assert Path(result.files["coordinates"]).name == "step5_input.crd"
    assert Path(result.files["toppar_str"]).name == "toppar.str"
    assert openmm_md.psf_file.endswith("step5_input.psf")
    assert openmm_md.crd_file.endswith("step5_input.crd")


def test_charmm_gui_backend_requires_token(tmp_path, monkeypatch):
    monkeypatch.delenv("CHARMMGUI_TOKEN", raising=False)
    monkeypatch.delenv("CHARMMGUI_EMAIL", raising=False)
    monkeypatch.delenv("CHARMMGUI_PASSWORD", raising=False)
    backend = CHARMMGUIBackend()
    md = _build_openmm_md(tmp_path)

    with pytest.raises(ValueError, match="requires either an API token or login credentials"):
        backend.prepare_model(
            md,
            str(tmp_path / "parameters"),
            pdb_reader_jobid="12345",
            quick_bilayer={"membtype": "PMm", "margin": 20},
        )


def test_charmm_gui_backend_requires_pdb_reader_jobid_or_login_for_auto_creation(tmp_path):
    backend = CHARMMGUIBackend(token="token")
    md = _build_openmm_md(tmp_path)

    with pytest.raises(ValueError, match="requires login credentials|pdb_reader_jobid"):
        backend.prepare_model(
            md,
            str(tmp_path / "parameters"),
            quick_bilayer={"membtype": "PMm", "margin": 20},
        )


def test_charmm_gui_backend_builds_numeric_protein_projection_query_values():
    backend = CHARMMGUIBackend()

    payload = backend._build_quick_bilayer_request(
        "12345",
        {
            "upper": "POPC:POPE=7:3",
            "lower": "POPC:POPE=7:3",
            "margin": 15.0,
            "prot_projection_upper": 1148.86415,
            "prot_projection_lower": 1810.1003,
            "ppm": False,
        },
    )

    assert payload["query"]["prot_projection_upper"] == "1148.86415"
    assert payload["query"]["prot_projection_lower"] == "1810.1003"
    assert payload["query"]["ppm"] == "false"


def test_resolve_request_mode_prefers_cached_module_in_collect_mode():
    backend = CHARMMGUIBackend()

    request_mode = backend._resolve_request_mode(
        options={"quick_bilayer": {"membtype": "PMm", "margin": 20}},
        model_name="modelA",
        cached_request_payload={"module": "membrane_builder"},
        submit_response={"module": "membrane_builder"},
        workflow_mode="collect",
    )

    assert request_mode == "membrane_builder"


def test_charmm_gui_backend_submits_polls_downloads_and_caches_inputs(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend(token="token", poll_interval_s=0.0, timeout_s=5.0)
    md = _build_openmm_md(tmp_path)
    parameters_folder = tmp_path / "parameters"

    requests = []
    download_bytes = _make_tarball(
        {
            "job/amber/step5_input.parm7": "PARM7",
            "job/amber/step5_input.rst7": "RST7",
            "job/openmm/system.xml": "<System/>",
        }
    )
    status_calls = {"count": 0}

    def fake_urlopen(request):
        requests.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "data": request.data.decode("utf-8") if request.data else None,
                "headers": dict(request.header_items()),
            }
        )
        if request.full_url.startswith("https://www.charmm-gui.org/api/quick_bilayer"):
            payload = {
                "jobid": "5804504324",
                "modules": "membuilder ffconverter",
                "submitted": "true",
            }
            return _FakeHTTPResponse(json.dumps(payload).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/check_status"):
            status_calls["count"] += 1
            status = "running quick_bilayer" if status_calls["count"] == 1 else "done"
            payload = {"status": status, "lastOutFile": "step5_assembly.out"}
            return _FakeHTTPResponse(json.dumps(payload).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/download"):
            return _FakeHTTPResponse(download_bytes)
        raise AssertionError(f"Unexpected URL: {request.full_url}")

    monkeypatch.setattr(charmm_gui_module, "_urlopen", fake_urlopen)

    result = backend.prepare_model(
        md,
        str(parameters_folder),
        pdb_reader_jobid={"modelA": "12345"},
        quick_bilayer={"membtype": "PMm", "margin": 20, "ppm": True},
    )

    assert isinstance(result, ParameterizationResult)
    assert result.prmtop_path.endswith("step5_input.parm7")
    assert result.coordinates_path.endswith("step5_input.rst7")
    assert md.prmtop_file == result.prmtop_path
    assert md.inpcrd_file == result.coordinates_path
    assert result.metadata["pdb_reader_jobid"] == "12345"
    assert result.metadata["quick_bilayer_jobid"] == "5804504324"

    model_root = parameters_folder / "modelA"
    assert (model_root / "request.json").exists()
    assert (model_root / "submit_response.json").exists()
    assert (model_root / "status_history.jsonl").exists()
    assert (model_root / "download.tgz").exists()
    assert (model_root / "selected_inputs.json").exists()

    submit_request = requests[0]
    assert submit_request["method"] == "POST"
    assert "jobid=12345" in submit_request["url"]
    assert "margin=20.0" in submit_request["url"]
    assert "run_ffconverter=true" in submit_request["url"]
    assert "ppm=true" in submit_request["url"]
    assert submit_request["data"] == "membtype=PMm"


def test_charmm_gui_backend_submit_only_returns_pending_result(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend(token="token", poll_interval_s=0.0, timeout_s=5.0)
    md = _build_openmm_md(tmp_path)
    parameters_folder = tmp_path / "parameters"

    requests = []

    def fake_urlopen(request):
        requests.append(request.full_url)
        if request.full_url.startswith("https://www.charmm-gui.org/api/quick_bilayer"):
            payload = {"jobid": "5804504324", "submitted": "true"}
            return _FakeHTTPResponse(json.dumps(payload).encode("utf-8"))
        raise AssertionError(f"Unexpected URL: {request.full_url}")

    monkeypatch.setattr(charmm_gui_module, "_urlopen", fake_urlopen)

    result = backend.prepare_model(
        md,
        str(parameters_folder),
        workflow_mode="submit_only",
        pdb_reader_jobid="12345",
        quick_bilayer={"membtype": "PMm", "margin": 20},
    )

    assert isinstance(result, ParameterizationResult)
    assert result.input_format == "remote_pending"
    assert result.metadata["state"] == "submitted"
    assert result.metadata["quick_bilayer_jobid"] == "5804504324"
    model_root = parameters_folder / "modelA"
    assert (model_root / "submit_response.json").exists()
    assert not (model_root / "download.tgz").exists()
    assert requests == [
        "https://www.charmm-gui.org/api/quick_bilayer?jobid=12345&margin=20.0&wdist=22.5&ion_conc=0.15&ion_type=NaCl&run_ffconverter=true"
    ]


def test_charmm_gui_backend_submit_only_can_fallback_to_site_quick_bilayer_submit(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend(
        email="user@example.org",
        password="secret",
        poll_interval_s=0.0,
        timeout_s=5.0,
    )
    md = _build_openmm_md(tmp_path)
    parameters_folder = tmp_path / "parameters"

    api_requests = []
    site_requests = []

    def fake_urlopen(request):
        api_requests.append(request.full_url)
        if request.full_url == "https://www.charmm-gui.org/api/login":
            return _FakeHTTPResponse(json.dumps({"token": "jwt-token"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/quick_bilayer"):
            payload = {
                "jobid": "12345",
                "modules": "membuilder ffconverter",
                "submitted": False,
            }
            return _FakeHTTPResponse(json.dumps(payload).encode("utf-8"))
        raise AssertionError(f"Unexpected API URL: {request.full_url}")

    def fake_site_open(request):
        site_requests.append(request.full_url)
        if request.full_url == "https://charmm-gui.org/?doc=sign":
            return _FakeHTTPResponse(b"logged-in")
        if request.full_url == "https://charmm-gui.org/?doc=input/membrane.quick&step=1":
            return _FakeHTTPResponse(_make_quick_site_submit_html("5804504999").encode("utf-8"))
        if request.full_url == "https://charmm-gui.org/?doc=input/retriever&jobid=5804504999":
            return _FakeHTTPResponse(
                _make_retriever_html("5804504999", include_quick_bilayer=True, quick_bilayer_jobid="5804504999").encode(
                    "utf-8"
                )
            )
        raise AssertionError(f"Unexpected site URL: {request.full_url}")

    monkeypatch.setattr(charmm_gui_module, "_urlopen", fake_urlopen)
    monkeypatch.setattr(charmm_gui_module, "_build_opener", lambda *args, **kwargs: _FakeSiteOpener(fake_site_open))

    result = backend.prepare_model(
        md,
        str(parameters_folder),
        workflow_mode="submit_only",
        pdb_reader_jobid="12345",
        quick_bilayer={"membtype": "PMm", "margin": 20},
    )

    assert isinstance(result, ParameterizationResult)
    assert result.input_format == "remote_pending"
    assert result.metadata["quick_bilayer_jobid"] == "5804504999"
    model_root = parameters_folder / "modelA"
    submit_response = json.loads((model_root / "submit_response.json").read_text())
    assert submit_response["source"] == "site"
    assert "https://charmm-gui.org/?doc=input/membrane.quick&step=1" in site_requests
    assert (model_root / "quick_bilayer_page.html").exists()
    assert api_requests[0] == "https://www.charmm-gui.org/api/login"


def test_charmm_gui_backend_membrane_builder_submit_only_returns_pending_result(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend(
        email="user@example.org",
        password="secret",
        poll_interval_s=0.0,
        timeout_s=5.0,
    )
    md = _build_openmm_md(tmp_path)
    parameters_folder = tmp_path / "parameters"

    api_requests = []
    site_requests = []

    def fake_urlopen(request):
        api_requests.append(request.full_url)
        raise AssertionError(f"Unexpected API URL: {request.full_url}")

    def fake_site_open(request):
        site_requests.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "data": request.data.decode("utf-8") if request.data else None,
            }
        )
        if request.full_url == "https://charmm-gui.org/?doc=sign":
            return _FakeHTTPResponse(b"logged-in")
        if request.full_url == "https://charmm-gui.org/?doc=input/membrane.bilayer&step=1&jobid=12345&project=membrane.quick":
            return _FakeHTTPResponse(_make_membrane_builder_step1_html("12345").encode("utf-8"))
        if request.full_url == "https://charmm-gui.org/?doc=input/membrane.bilayer.size":
            body = request.data.decode("utf-8")
            assert "lipid_ratio%5Bupper%5D%5Bpopc%5D=7" in body
            assert "lipid_ratio%5Blower%5D%5Bpope%5D=3" in body
            return _FakeHTTPResponse(
                b"<div>Calculated</div>",
                headers={
                    "X-JSON": json.dumps(
                        {
                            "upper": {"popc": 350, "pope": 150},
                            "lower": {"popc": 350, "pope": 150},
                            "area": {"popc": "68.3", "pope": "58.8"},
                            "valid": True,
                            "area_updated": "size",
                        }
                    )
                },
            )
        if request.full_url == "https://charmm-gui.org/?doc=input/membrane.bilayer&step=3":
            body = request.data.decode("utf-8")
            assert "hetero_lx=180.0" in body
            assert "hetero_wdist=22.5" in body
            assert "lipid_ratio%5Bupper%5D%5Bpopc%5D=7" in body
            assert "lipid_ratio%5Bupper%5D%5Bpope%5D=3" in body
            assert "lipid_ratio%5Blower%5D%5Bpopc%5D=7" in body
            assert "lipid_ratio%5Blower%5D%5Bpope%5D=3" in body
            assert "lipid_number%5Bupper%5D%5Bpopc%5D=350" in body
            assert "lipid_number%5Blower%5D%5Bpope%5D=150" in body
            assert "check_penetration=1" in body
            return _FakeHTTPResponse(b"<html><body>JOB ID: 12345</body></html>")
        if request.full_url == "https://charmm-gui.org/?doc=input/retriever&jobid=12345":
            return _FakeHTTPResponse(
                """
                <html><body>
                <table id="recovery_table">
                  <tr>
                    <td>membrane.bilayer</td>
                    <td>4</td>
                    <td><a href="?doc=input/membrane.bilayer&step=4&project=pdbreader&jobid=12345">Go</a></td>
                  </tr>
                </table>
                </body></html>
                """.encode("utf-8")
            )
        raise AssertionError(f"Unexpected site URL: {request.full_url}")

    monkeypatch.setattr(charmm_gui_module, "_urlopen", fake_urlopen)
    monkeypatch.setattr(charmm_gui_module, "_build_opener", lambda *args, **kwargs: _FakeSiteOpener(fake_site_open))

    result = backend.prepare_model(
        md,
        str(parameters_folder),
        workflow_mode="submit_only",
        pdb_reader_jobid="12345",
        membrane_builder={
            "upper": "POPC=7:POPE=3",
            "lower": "POPC=7:POPE=3",
            "lateral_length": 180,
            "wdist": 22.5,
            "source_project": "membrane.quick",
        },
    )

    assert result.input_format == "remote_pending"
    assert result.metadata["state"] == "submitted"
    assert result.metadata["module"] == "membrane_builder"
    assert result.metadata["membrane_builder_jobid"] == "12345"
    model_root = parameters_folder / "modelA"
    request_json = json.loads((model_root / "request.json").read_text())
    assert request_json["module"] == "membrane_builder"
    assert request_json["query"]["project"] == "membrane.quick"
    assert json.loads((model_root / "submit_response.json").read_text())["module"] == "membrane_builder"
    assert (model_root / "membrane_builder_step1.html").exists()
    assert (model_root / "membrane_builder_submit.html").exists()
    assert (model_root / "membrane_builder_manifest.json").exists()
    assert api_requests == []
    assert any(
        request["url"] == "https://charmm-gui.org/?doc=input/membrane.bilayer&step=1&jobid=12345&project=membrane.quick"
        for request in site_requests
    )


def test_submit_membrane_builder_recovers_when_submit_post_times_out(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend(membrane_builder_submit_timeout_s=7.5)

    submit_calls = []
    autosubmit_calls = []

    monkeypatch.setattr(
        backend,
        "_site_request",
        lambda session, url, method="GET", data=None, headers=None, timeout_s=None: _make_membrane_builder_step1_html(
            "12345"
        ).encode("utf-8"),
    )
    monkeypatch.setattr(
        backend,
        "_apply_membrane_builder_size_calculation",
        lambda session, site_base_url, fields: list(fields),
    )

    def fake_submit_form(session, site_base_url, action, method, fields, timeout_s=None):
        submit_calls.append(
            {
                "action": action,
                "method": method,
                "timeout_s": timeout_s,
            }
        )
        raise TimeoutError("submit stalled")

    monkeypatch.setattr(backend, "_site_submit_form", fake_submit_form)
    monkeypatch.setattr(
        backend,
        "_recover_membrane_builder_submit_timeout",
        lambda session, site_base_url, jobid, project, submit_snapshot_path: "<html><body>JOB ID: 12345</body></html>",
    )

    def fake_resolve_site_autosubmit_forms(session, site_base_url, html_text, max_hops=4, submit_timeout_s=None):
        autosubmit_calls.append({"submit_timeout_s": submit_timeout_s, "html_text": html_text})
        return html_text

    monkeypatch.setattr(backend, "_resolve_site_autosubmit_forms", fake_resolve_site_autosubmit_forms)

    request_payload, submit_response = backend._submit_membrane_builder(
        session=object(),
        site_base_url="https://charmm-gui.org",
        source_jobid="12345",
        source_project="membrane.quick",
        membrane_builder={
            "upper": "POPC=7:POPE=3",
            "lower": "POPC=7:POPE=3",
            "lateral_length": 150,
            "wdist": 22.5,
        },
        step1_snapshot_path=tmp_path / "builder_step1.html",
        submit_snapshot_path=tmp_path / "builder_submit.html",
    )

    assert submit_calls == [
        {
            "action": "./?doc=input/membrane.bilayer&step=3",
            "method": "POST",
            "timeout_s": 7.5,
        }
    ]
    assert autosubmit_calls == [{"submit_timeout_s": 7.5, "html_text": "<html><body>JOB ID: 12345</body></html>"}]
    assert request_payload["module"] == "membrane_builder"
    assert submit_response["jobid"] == "12345"
    assert (tmp_path / "builder_step1.html").exists()
    assert (tmp_path / "builder_submit.html").exists()


def test_load_membrane_builder_page_state_parses_msg_form(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend()

    html = (
        "<body><form name='msg_form' method=post action='./?doc=input/membrane.bilayer&step=3'>"
        "<input type=hidden name='jobid' value='12345'>"
        "<input type=hidden name='project' value='membrane.quick'>"
        "<input type=hidden name='rollback' value='1'>"
        "</form></body><script>document.msg_form.submit();</script>"
    )

    monkeypatch.setattr(
        backend,
        "_site_request",
        lambda session, url, method="GET", data=None, headers=None, timeout_s=None: html.encode("utf-8"),
    )

    page_state = backend._load_membrane_builder_page_state(
        session=object(),
        site_base_url="https://charmm-gui.org",
        jobid="12345",
        project="membrane.quick",
        step="3",
        snapshot_path=tmp_path / "builder_step3.html",
    )

    assert page_state["action"] == "./?doc=input/membrane.bilayer&step=3"
    assert page_state["method"] == "POST"
    assert ("jobid", "12345") in page_state["fields"]
    assert ("project", "membrane.quick") in page_state["fields"]


def test_resolve_site_autosubmit_forms_follows_msg_form(monkeypatch):
    backend = CHARMMGUIBackend()
    submit_html = (
        "<body><form name='msg_form' method=post action='./?doc=input/membrane.bilayer&step=3'>"
        "<input type=hidden name='jobid' value='12345'>"
        "<input type=hidden name='project' value='membrane.quick'>"
        "</form></body><script>document.msg_form.submit();</script>"
    )
    final_html = "<html><body><div>converter.ffconverter</div></body></html>"

    monkeypatch.setattr(
        backend,
        "_site_submit_form",
        lambda session, site_base_url, action, method, fields, timeout_s=None: final_html,
    )

    resolved = backend._resolve_site_autosubmit_forms(
        session=object(),
        site_base_url="https://charmm-gui.org",
        html_text=submit_html,
    )

    assert resolved == final_html


def test_load_direct_ffconverter_step_returns_step_one(monkeypatch):
    backend = CHARMMGUIBackend()
    html = """
    <html>
      <body>
        <form id='ffconverter' method='post' action='./?doc=input/converter.ffconverter&step=2'>
          <input type='hidden' name='jobid' value='12345'>
          <input type='hidden' name='project' value='membrane.bilayer'>
          <input type='hidden' name='systype' value='solution'>
          <input type='hidden' name='fftype' value='c36m'>
        </form>
      </body>
    </html>
    """

    monkeypatch.setattr(
        backend,
        "_site_request",
        lambda session, url, method="GET", data=None, headers=None: html.encode("utf-8"),
    )

    step = backend._load_direct_ffconverter_step(
        session=object(),
        site_base_url="https://charmm-gui.org",
        jobid="12345",
        project="membrane.bilayer",
    )

    assert step is not None
    assert step["doc"] == "converter.ffconverter"
    assert step["jobid"] == "12345"
    assert step["project"] == "membrane.bilayer"
    assert step["step"] == "1"


def test_extract_ffconverter_step_from_retriever_filters_by_project():
    backend = CHARMMGUIBackend()
    html = _make_retriever_html(
        jobid="12345",
        include_ffconverter=True,
        ffconverter_jobid="12345",
        ffconverter_step="6",
        ffconverter_project="pdbreader",
    )
    html += _make_retriever_html(
        jobid="12345",
        include_ffconverter=True,
        ffconverter_jobid="12345",
        ffconverter_step="2",
        ffconverter_project="membrane.bilayer",
    )

    default_step = backend._extract_ffconverter_step_from_retriever(html)
    builder_step = backend._extract_ffconverter_step_from_retriever(html, expected_project="membrane.bilayer")

    assert default_step is not None
    assert default_step["project"] == "pdbreader"
    assert builder_step is not None
    assert builder_step["project"] == "membrane.bilayer"


def test_extract_membrane_builder_step_from_retriever_uses_latest_step():
    backend = CHARMMGUIBackend()
    html = """
    <html><body>
    <table id="recovery_table">
      <tr>
        <td>membrane.bilayer</td>
        <td>3</td>
        <td><a href="?doc=input/membrane.bilayer&step=3&project=membrane.quick&jobid=12345">Go</a></td>
      </tr>
      <tr>
        <td>membrane.bilayer</td>
        <td>4</td>
        <td><a href="?doc=input/membrane.bilayer&step=4&project=membrane.quick&jobid=12345">Go</a></td>
      </tr>
    </table>
    </body></html>
    """

    step = backend._extract_membrane_builder_step_from_retriever(html)

    assert step is not None
    assert step["step"] == "4"


def test_poll_until_done_treats_compressing_as_running(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend()
    statuses = iter(
        [
            {
                "jobid": "12345",
                "status": "compressing",
                "hasTarFile": False,
                "lastInpStep": "7",
            },
            {
                "jobid": "12345",
                "status": "done",
                "hasTarFile": True,
                "lastInpStep": "7",
            },
        ]
    )

    monkeypatch.setattr(backend, "_check_status", lambda api_base_url, token, jobid: next(statuses))
    monkeypatch.setattr(charmm_gui_module.time, "sleep", lambda seconds: None)

    result = backend._poll_until_done(
        api_base_url="https://example.org/api",
        token="token",
        jobid="12345",
        status_history_path=tmp_path / "status_history.jsonl",
        poll_interval_s=0.0,
        timeout_s=1.0,
    )

    assert result["status"] == "done"
    lines = (tmp_path / "status_history.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2


def test_poll_until_done_treats_site_archive_ready_as_done(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend()
    statuses = iter(
        [
            {
                "jobid": "12345",
                "status": "compressing",
                "hasTarFile": False,
                "lastInpStep": "7",
            },
        ]
    )

    monkeypatch.setattr(backend, "_check_status", lambda api_base_url, token, jobid: next(statuses))
    monkeypatch.setattr(backend, "_create_site_session", lambda site_base_url, email, password: object())
    monkeypatch.setattr(backend, "_get_site_archive_compression_progress", lambda *args, **kwargs: 100.0)
    monkeypatch.setattr(charmm_gui_module.time, "sleep", lambda seconds: None)

    result = backend._poll_until_done(
        api_base_url="https://example.org/api",
        token="token",
        jobid="12345",
        status_history_path=tmp_path / "status_history.jsonl",
        poll_interval_s=0.0,
        timeout_s=1.0,
        site_base_url="https://charmm-gui.org",
        email="user@example.org",
        password="secret",
    )

    assert result["status"] == "done"
    assert result["source"] == "site_archive_ready"
    lines = (tmp_path / "status_history.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2


def test_advance_membrane_builder_to_ffconverter_uses_direct_ffconverter_when_builder_done(
    tmp_path, monkeypatch
):
    backend = CHARMMGUIBackend()

    monkeypatch.setattr(
        backend,
        "_site_request",
        lambda session, url, method="GET", data=None, headers=None: _make_retriever_html(
            include_membrane_builder=True,
            membrane_builder_step="4",
            membrane_builder_project="membrane.quick",
        ).encode("utf-8"),
    )
    monkeypatch.setattr(
        backend,
        "_get_jobids_page_status",
        lambda *args, **kwargs: {"status": "Done"},
    )
    monkeypatch.setattr(
        backend,
        "_load_direct_ffconverter_step",
        lambda *args, **kwargs: {
            "doc": "converter.ffconverter",
            "jobid": "12345",
            "project": "membrane.bilayer",
            "step": "1",
        },
    )

    status = backend._advance_membrane_builder_to_ffconverter(
        session=object(),
        site_base_url="https://charmm-gui.org",
        model_root=tmp_path,
        membrane_builder_jobid="12345",
        source_project="membrane.quick",
        retriever_snapshot_path=tmp_path / "retriever.html",
        jobids_snapshot_path=tmp_path / "jobids.html",
        progress_snapshot_path=tmp_path / "progress.html",
    )

    assert status["state"] == "ffconverter_ready"
    assert status["ffconverter_step"]["jobid"] == "12345"


def test_download_archive_via_site_skips_compress_only_when_archive_is_already_ready(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend()
    destination = tmp_path / "download.tgz"
    download_bytes = _make_tarball({"job/amber/step5_input.parm7": "PARM7"})
    requests = []

    class _Session:
        def open(self, request, timeout=None):
            requests.append(request.full_url)
            if request.full_url == "https://charmm-gui.org/?doc=input/download&jobid=12345":
                return _FakeHTTPResponse(download_bytes)
            raise AssertionError(f"Unexpected download URL: {request.full_url}")

    monkeypatch.setattr(backend, "_create_site_session", lambda *args, **kwargs: _Session())
    monkeypatch.setattr(backend, "_get_site_archive_compression_progress", lambda *args, **kwargs: 100.0)
    monkeypatch.setattr(
        backend,
        "_site_request",
        lambda session, url, method="GET", data=None, headers=None: (_ for _ in ()).throw(
            AssertionError(f"compress_only should not be called when archive is already ready: {url}")
        ),
    )

    backend._download_archive_via_site(
        site_base_url="https://charmm-gui.org",
        email="user@example.org",
        password="secret",
        jobid="12345",
        destination=destination,
        poll_interval_s=0.0,
        timeout_s=1.0,
    )

    assert destination.exists()
    assert requests == ["https://charmm-gui.org/?doc=input/download&jobid=12345"]


def test_charmm_gui_backend_can_login_and_use_returned_token(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend(
        email="user@example.org",
        password="secret",
        poll_interval_s=0.0,
        timeout_s=5.0,
        verify_quick_bilayer_submission=False,
    )
    md = _build_openmm_md(tmp_path)
    parameters_folder = tmp_path / "parameters"

    requests = []
    download_bytes = _make_tarball(
        {
            "job/amber/step5_input.parm7": "PARM7",
            "job/amber/step5_input.rst7": "RST7",
        }
    )

    def fake_urlopen(request):
        request_info = {
            "url": request.full_url,
            "method": request.get_method(),
            "data": request.data.decode("utf-8") if request.data else None,
            "headers": dict(request.header_items()),
        }
        requests.append(request_info)
        if request.full_url == "https://www.charmm-gui.org/api/login":
            return _FakeHTTPResponse(json.dumps({"token": "jwt-token"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/quick_bilayer"):
            return _FakeHTTPResponse(json.dumps({"jobid": "5804504324", "submitted": "true"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/check_status"):
            return _FakeHTTPResponse(json.dumps({"status": "done"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/download"):
            return _FakeHTTPResponse(download_bytes)
        raise AssertionError(f"Unexpected URL: {request.full_url}")

    monkeypatch.setattr(charmm_gui_module, "_urlopen", fake_urlopen)

    result = backend.prepare_model(
        md,
        str(parameters_folder),
        pdb_reader_jobid="12345",
        quick_bilayer={"membtype": "PMm", "margin": 20},
    )

    assert result.prmtop_path.endswith("step5_input.parm7")
    assert requests[0]["url"] == "https://www.charmm-gui.org/api/login"
    assert requests[0]["method"] == "POST"
    assert requests[0]["data"] == json.dumps({"email": "user@example.org", "password": "secret"})
    assert "Authorization" not in requests[0]["headers"]
    assert requests[1]["headers"]["Authorization"] == "Bearer jwt-token"


def test_charmm_gui_backend_can_create_pdb_reader_job_from_local_input(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend(email="user@example.org", password="secret", poll_interval_s=0.0, timeout_s=5.0)
    md = _build_openmm_md(tmp_path)
    parameters_folder = tmp_path / "parameters"

    download_bytes = _make_tarball(
        {
            "job/amber/step5_input.parm7": "PARM7",
            "job/amber/step5_input.rst7": "RST7",
        }
    )
    api_requests = []
    site_requests = []

    def fake_urlopen(request):
        api_requests.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "data": request.data.decode("utf-8") if request.data else None,
                "headers": dict(request.header_items()),
            }
        )
        if request.full_url == "https://www.charmm-gui.org/api/login":
            return _FakeHTTPResponse(json.dumps({"token": "jwt-token"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/quick_bilayer"):
            return _FakeHTTPResponse(json.dumps({"jobid": "5804504324", "submitted": "true"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/check_status"):
            return _FakeHTTPResponse(json.dumps({"status": "done"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/download"):
            return _FakeHTTPResponse(download_bytes)
        raise AssertionError(f"Unexpected URL: {request.full_url}")

    def fake_site_open(request):
        site_requests.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "data": request.data,
                "headers": dict(request.header_items()),
            }
        )
        if request.full_url == "https://charmm-gui.org/?doc=sign":
            return _FakeHTTPResponse(b"logged-in")
        if request.full_url == "https://charmm-gui.org/?doc=input/pdbreader&step=1":
            assert request.get_method() == "POST"
            assert "multipart/form-data" in dict(request.header_items())["Content-type"]
            return _FakeHTTPResponse(_make_pdb_reader_chain_selection_html().encode("utf-8"))
        if request.full_url == "https://charmm-gui.org/?doc=input/pdbreader&step=2":
            body = request.data.decode("utf-8")
            assert "chains%5BPROA%5D%5Bchecked%5D=1" in body
            assert "chains%5BHETO%5D" not in body
            return _FakeHTTPResponse(_make_pdb_reader_options_html().encode("utf-8"))
        if request.full_url == "https://charmm-gui.org/?doc=input/pdbreader&step=3":
            body = request.data.decode("utf-8")
            assert "system_pH=6.5" in body
            assert "hbuild_checked=on" in body
            return _FakeHTTPResponse(_make_pdb_reader_ready_html().encode("utf-8"))
        if request.full_url == "https://charmm-gui.org/?doc=input/retriever&jobid=12345":
            return _FakeHTTPResponse(
                _make_retriever_html("12345", include_quick_bilayer=True, quick_bilayer_jobid="5804504324").encode(
                    "utf-8"
                )
            )
        if request.full_url == "https://charmm-gui.org/?doc=input/retriever&jobid=5804504324":
            return _FakeHTTPResponse(
                _make_retriever_html("5804504324", include_ffconverter=True, ffconverter_step="6").encode("utf-8")
            )
        if request.full_url == "https://charmm-gui.org/?doc=input/membrane.quick&jobid=5804504324&project=pdbreader":
            return _FakeHTTPResponse(b"<html><body>not ready</body></html>")
        raise AssertionError(f"Unexpected site URL: {request.full_url}")

    monkeypatch.setattr(charmm_gui_module, "_urlopen", fake_urlopen)
    monkeypatch.setattr(charmm_gui_module, "_build_opener", lambda *args, **kwargs: _FakeSiteOpener(fake_site_open))

    result = backend.prepare_model(
        md,
        str(parameters_folder),
        pdb_reader={"system_pH": 6.5, "preserve_hydrogens": True},
        quick_bilayer={"membtype": "PMm", "margin": 20},
    )

    assert isinstance(result, ParameterizationResult)
    assert result.metadata["pdb_reader_jobid"] == "12345"
    assert result.metadata["quick_bilayer_jobid"] == "5804504324"
    assert (parameters_folder / "modelA" / "pdb_reader_manifest.json").exists()
    assert (parameters_folder / "modelA" / "quick_bilayer_manifest.json").exists()
    assert (parameters_folder / "modelA" / "pdb_reader_step1_chain_selection.html").exists()
    assert (parameters_folder / "modelA" / "pdb_reader_step2_options.html").exists()
    assert (parameters_folder / "modelA" / "pdb_reader_step3_reader.html").exists()
    assert any(request["url"] == "https://charmm-gui.org/?doc=input/pdbreader&step=1" for request in site_requests)
    assert "jobid=12345" in api_requests[1]["url"]


def test_charmm_gui_backend_recovers_from_pdb_reader_step3_timeout(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend(email="user@example.org", password="secret", poll_interval_s=0.0, timeout_s=5.0)
    md = _build_openmm_md(tmp_path)
    parameters_folder = tmp_path / "parameters"

    download_bytes = _make_tarball(
        {
            "job/amber/step5_input.parm7": "PARM7",
            "job/amber/step5_input.rst7": "RST7",
        }
    )
    site_requests = []

    def fake_urlopen(request):
        if request.full_url == "https://www.charmm-gui.org/api/login":
            return _FakeHTTPResponse(json.dumps({"token": "jwt-token"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/quick_bilayer"):
            return _FakeHTTPResponse(json.dumps({"jobid": "5804504324", "submitted": "true"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/check_status"):
            return _FakeHTTPResponse(json.dumps({"status": "done"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/download"):
            return _FakeHTTPResponse(download_bytes)
        raise AssertionError(f"Unexpected URL: {request.full_url}")

    def fake_site_open(request):
        site_requests.append(request.full_url)
        if request.full_url == "https://charmm-gui.org/?doc=sign":
            return _FakeHTTPResponse(b"logged-in")
        if request.full_url == "https://charmm-gui.org/?doc=input/pdbreader&step=1":
            return _FakeHTTPResponse(_make_pdb_reader_chain_selection_html().encode("utf-8"))
        if request.full_url == "https://charmm-gui.org/?doc=input/pdbreader&step=2":
            return _FakeHTTPResponse(_make_pdb_reader_options_html().encode("utf-8"))
        if request.full_url == "https://charmm-gui.org/?doc=input/pdbreader&step=3":
            raise TimeoutError("slow site submit")
        if request.full_url == "https://charmm-gui.org/?doc=input/pdbreader&jobid=12345&project=pdbreader":
            return _FakeHTTPResponse(_make_pdb_reader_ready_html().encode("utf-8"))
        if request.full_url == "https://charmm-gui.org/?doc=input/retriever&jobid=12345":
            return _FakeHTTPResponse(
                _make_retriever_html("12345", include_quick_bilayer=True, quick_bilayer_jobid="5804504324").encode(
                    "utf-8"
                )
            )
        if request.full_url == "https://charmm-gui.org/?doc=input/retriever&jobid=5804504324":
            return _FakeHTTPResponse(
                _make_retriever_html("5804504324", include_ffconverter=True, ffconverter_step="6").encode("utf-8")
            )
        if request.full_url == "https://charmm-gui.org/?doc=input/membrane.quick&jobid=5804504324&project=pdbreader":
            return _FakeHTTPResponse(b"<html><body>not ready</body></html>")
        raise AssertionError(f"Unexpected site URL: {request.full_url}")

    monkeypatch.setattr(charmm_gui_module, "_urlopen", fake_urlopen)
    monkeypatch.setattr(charmm_gui_module, "_build_opener", lambda *args, **kwargs: _FakeSiteOpener(fake_site_open))

    result = backend.prepare_model(
        md,
        str(parameters_folder),
        pdb_reader={"system_pH": 6.5, "preserve_hydrogens": True},
        quick_bilayer={"membtype": "PMm", "margin": 20},
    )

    assert isinstance(result, ParameterizationResult)
    assert result.metadata["pdb_reader_jobid"] == "12345"
    assert "https://charmm-gui.org/?doc=input/pdbreader&jobid=12345&project=pdbreader" in site_requests
    assert (parameters_folder / "modelA" / "pdb_reader_step3_reader.html").exists()


def test_charmm_gui_backend_raises_on_pdb_reader_mismatch_page(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend(email="user@example.org", password="secret", poll_interval_s=0.0, timeout_s=5.0)
    md = _build_openmm_md(tmp_path)

    def fake_urlopen(request):
        if request.full_url == "https://www.charmm-gui.org/api/login":
            return _FakeHTTPResponse(json.dumps({"token": "jwt-token"}).encode("utf-8"))
        raise AssertionError(f"API should not be called after PDB Reader mismatch: {request.full_url}")

    def fake_site_open(request):
        if request.full_url == "https://charmm-gui.org/?doc=sign":
            return _FakeHTTPResponse(b"logged-in")
        if request.full_url == "https://charmm-gui.org/?doc=input/pdbreader&step=1":
            return _FakeHTTPResponse(_make_pdb_reader_chain_selection_html().encode("utf-8"))
        if request.full_url == "https://charmm-gui.org/?doc=input/pdbreader&step=2":
            return _FakeHTTPResponse(_make_pdb_reader_options_html().encode("utf-8"))
        if request.full_url == "https://charmm-gui.org/?doc=input/pdbreader&step=3":
            html = "Mismatch in Ligand Atom Order<script>var step = 'reader';</script>"
            return _FakeHTTPResponse(html.encode("utf-8"))
        raise AssertionError(f"Unexpected site URL: {request.full_url}")

    monkeypatch.setattr(charmm_gui_module, "_urlopen", fake_urlopen)
    monkeypatch.setattr(charmm_gui_module, "_build_opener", lambda *args, **kwargs: _FakeSiteOpener(fake_site_open))

    with pytest.raises(RuntimeError, match="PDB Reader"):
        backend.prepare_model(
            md,
            str(tmp_path / "parameters"),
            quick_bilayer={"membtype": "PMm", "margin": 20},
        )


def test_charmm_gui_backend_raises_on_error_status(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend(token="token", poll_interval_s=0.0, timeout_s=1.0)
    md = _build_openmm_md(tmp_path)

    def fake_urlopen(request):
        if request.full_url.startswith("https://www.charmm-gui.org/api/quick_bilayer"):
            return _FakeHTTPResponse(json.dumps({"jobid": "999", "submitted": "true"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/check_status"):
            payload = {"status": "error", "lastOutLines": "builder failed"}
            return _FakeHTTPResponse(json.dumps(payload).encode("utf-8"))
        raise AssertionError(f"Unexpected URL: {request.full_url}")

    monkeypatch.setattr(charmm_gui_module, "_urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="builder failed"):
        backend.prepare_model(
            md,
            str(tmp_path / "parameters"),
            pdb_reader_jobid="12345",
            quick_bilayer={"membtype": "PMm", "margin": 20},
        )


def test_charmm_gui_backend_raises_on_error_like_status(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend(token="token", poll_interval_s=0.0, timeout_s=1.0)
    md = _build_openmm_md(tmp_path)

    def fake_urlopen(request):
        if request.full_url.startswith("https://www.charmm-gui.org/api/quick_bilayer"):
            return _FakeHTTPResponse(json.dumps({"jobid": "999", "submitted": "true"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/check_status"):
            payload = {"status": "error occurred from quick_bilayer"}
            return _FakeHTTPResponse(json.dumps(payload).encode("utf-8"))
        raise AssertionError(f"Unexpected URL: {request.full_url}")

    monkeypatch.setattr(charmm_gui_module, "_urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="quick_bilayer"):
        backend.prepare_model(
            md,
            str(tmp_path / "parameters"),
            pdb_reader_jobid="12345",
            quick_bilayer={"membtype": "PMm", "margin": 20},
        )


def test_charmm_gui_backend_can_reuse_cached_inputs_without_network(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend(token="token")
    md = _build_openmm_md(tmp_path)
    parameters_folder = tmp_path / "parameters"
    model_root = parameters_folder / "modelA"
    extract_root = model_root / "extracted" / "amber"
    extract_root.mkdir(parents=True, exist_ok=True)
    topology_path = extract_root / "step5_input.parm7"
    coordinates_path = extract_root / "step5_input.rst7"
    topology_path.write_text("PARM7")
    coordinates_path.write_text("RST7")
    (model_root / "selected_inputs.json").write_text(
        json.dumps(
            {
                "input_format": "amber",
                "files": {
                    "prmtop": str(topology_path.relative_to(model_root)),
                    "coordinates": str(coordinates_path.relative_to(model_root)),
                },
                "metadata": {
                    "backend": "charmm_gui",
                    "quick_bilayer_jobid": "5804504324",
                },
            }
        )
    )

    def fail_urlopen(request):
        raise AssertionError(f"Network should not be used when cache is available: {request.full_url}")

    monkeypatch.setattr(charmm_gui_module, "_urlopen", fail_urlopen)

    result = backend.prepare_model(
        md,
        str(parameters_folder),
        allow_remote=False,
        pdb_reader_jobid="12345",
        quick_bilayer={"membtype": "PMm", "margin": 20},
    )

    assert result.prmtop_path == str(topology_path.resolve())
    assert result.coordinates_path == str(coordinates_path.resolve())


def test_charmm_gui_backend_collect_mode_uses_cached_submission_metadata(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend(token="token", poll_interval_s=0.0, timeout_s=5.0)
    md = _build_openmm_md(tmp_path)
    parameters_folder = tmp_path / "parameters"
    model_root = parameters_folder / "modelA"
    model_root.mkdir(parents=True, exist_ok=True)
    cached_request = {
        "query": {
            "jobid": "12345",
            "margin": "15.0",
            "wdist": "22.5",
            "ion_conc": "0.15",
            "ion_type": "NaCl",
            "ppm": "false",
            "run_ffconverter": "true",
        },
        "form": {
            "upper": "POPC:POPE=7:3",
            "lower": "POPC:POPE=7:3",
        },
    }
    (model_root / "request.json").write_text(json.dumps(cached_request))
    (model_root / "submit_response.json").write_text(json.dumps({"jobid": "5804504324", "submitted": True}))

    requests = []
    download_bytes = _make_tarball(
        {
            "job/amber/step5_input.parm7": "PARM7",
            "job/amber/step5_input.rst7": "RST7",
        }
    )

    def fake_urlopen(request):
        requests.append(request.full_url)
        if request.full_url.startswith("https://www.charmm-gui.org/api/check_status"):
            return _FakeHTTPResponse(json.dumps({"status": "done"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/download"):
            return _FakeHTTPResponse(download_bytes)
        raise AssertionError(f"Unexpected URL: {request.full_url}")

    monkeypatch.setattr(charmm_gui_module, "_urlopen", fake_urlopen)

    result = backend.prepare_model(
        md,
        str(parameters_folder),
        workflow_mode="collect",
    )

    assert result.prmtop_path.endswith("step5_input.parm7")
    assert result.metadata["pdb_reader_jobid"] == "12345"
    assert result.metadata["quick_bilayer_jobid"] == "5804504324"
    assert result.metadata["request"] == cached_request
    assert json.loads((model_root / "request.json").read_text()) == cached_request
    assert requests == [
        "https://www.charmm-gui.org/api/check_status?jobid=5804504324",
        "https://www.charmm-gui.org/api/download?jobid=5804504324",
    ]


def test_charmm_gui_backend_collect_mode_uses_jobids_page_done_when_api_is_stale(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend(
        token="token",
        email="user@example.org",
        password="secret",
        poll_interval_s=0.0,
        timeout_s=5.0,
        prefer_site_download=False,
    )
    md = _build_openmm_md(tmp_path)
    parameters_folder = tmp_path / "parameters"
    model_root = parameters_folder / "modelA"
    model_root.mkdir(parents=True, exist_ok=True)
    cached_request = {
        "query": {
            "jobid": "12345",
            "margin": "15.0",
            "wdist": "22.5",
            "ion_conc": "0.15",
            "ion_type": "NaCl",
            "ppm": "false",
            "run_ffconverter": "true",
        },
        "form": {
            "upper": "POPC:POPE=7:3",
            "lower": "POPC:POPE=7:3",
        },
    }
    (model_root / "request.json").write_text(json.dumps(cached_request))
    (model_root / "submit_response.json").write_text(json.dumps({"jobid": "5804504324", "submitted": True}))
    (model_root / "quick_bilayer_manifest.json").write_text(
        json.dumps({"jobid": "5804504324", "step": "3", "doc": "membrane.quick", "project": "pdbreader"})
    )

    api_requests = []
    download_bytes = _make_tarball(
        {
            "job/amber/step5_input.parm7": "PARM7",
            "job/amber/step5_input.rst7": "RST7",
        }
    )

    def fake_urlopen(request):
        api_requests.append(request.full_url)
        if request.full_url.startswith("https://www.charmm-gui.org/api/check_status"):
            return _FakeHTTPResponse(json.dumps({"status": "submitted", "hasTarFile": False}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/download"):
            raise AssertionError("API download should not be used when Job IDs page already says Done.")
        raise AssertionError(f"Unexpected URL: {request.full_url}")

    def fake_site_open(request):
        if request.full_url == "https://charmm-gui.org/?doc=sign":
            return _FakeHTTPResponse(b"logged-in")
        if request.full_url == "https://charmm-gui.org/?doc=input/retriever&jobid=5804504324":
            return _FakeHTTPResponse(
                _make_retriever_html("5804504324", include_ffconverter=True, ffconverter_step="6").encode("utf-8")
            )
        if request.full_url == "https://charmm-gui.org/?doc=user/jobids":
            return _FakeHTTPResponse(
                _make_jobids_html(
                    [
                        {
                            "jobid": "5804504324",
                            "project": "PDB Reader & Manipulator",
                            "module": "Quick Bilayer",
                            "step": "3",
                            "status": "Done",
                        }
                    ]
                ).encode("utf-8")
            )
        if request.full_url == "https://charmm-gui.org/?doc=input/download&jobid=5804504324&compress_only=1":
            return _FakeHTTPResponse(b"0")
        if request.full_url == "https://charmm-gui.org/?doc=input/download&jobid=5804504324&check_only=1":
            return _FakeHTTPResponse(b"100")
        if request.full_url == "https://charmm-gui.org/?doc=input/download&jobid=5804504324":
            return _FakeHTTPResponse(download_bytes)
        raise AssertionError(f"Unexpected site URL: {request.full_url}")

    monkeypatch.setattr(charmm_gui_module, "_urlopen", fake_urlopen)
    monkeypatch.setattr(charmm_gui_module, "_build_opener", lambda *args, **kwargs: _FakeSiteOpener(fake_site_open))

    result = backend.prepare_model(
        md,
        str(parameters_folder),
        workflow_mode="collect",
    )

    assert result.prmtop_path.endswith("step5_input.parm7")
    assert result.metadata["quick_bilayer_jobid"] == "5804504324"
    final_status = json.loads((model_root / "final_status.json").read_text())
    assert final_status["source"] == "jobids_page"
    assert final_status["status"] == "Done"
    assert api_requests == [
        "https://www.charmm-gui.org/api/check_status?jobid=5804504324",
    ]


def test_charmm_gui_backend_collect_mode_returns_pending_when_ffconverter_not_yet_exposed(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend(
        token="token",
        email="user@example.org",
        password="secret",
        poll_interval_s=0.0,
        timeout_s=5.0,
        prefer_site_download=False,
    )
    md = _build_openmm_md(tmp_path)
    parameters_folder = tmp_path / "parameters"
    model_root = parameters_folder / "modelA"
    model_root.mkdir(parents=True, exist_ok=True)
    cached_request = {
        "query": {
            "jobid": "12345",
            "margin": "15.0",
            "wdist": "22.5",
            "ion_conc": "0.15",
            "ion_type": "NaCl",
            "ppm": "false",
            "run_ffconverter": "true",
        },
        "form": {
            "upper": "POPC:POPE=7:3",
            "lower": "POPC:POPE=7:3",
        },
    }
    (model_root / "request.json").write_text(json.dumps(cached_request))
    (model_root / "submit_response.json").write_text(json.dumps({"jobid": "5804504324", "submitted": True}))
    (model_root / "quick_bilayer_manifest.json").write_text(
        json.dumps({"jobid": "5804504324", "step": "3", "doc": "membrane.quick", "project": "pdbreader"})
    )

    def fake_urlopen(request):
        if request.full_url.startswith("https://www.charmm-gui.org/api/check_status"):
            return _FakeHTTPResponse(json.dumps({"status": "submitted", "hasTarFile": False}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/download"):
            raise AssertionError("Archive download should not start before FF-Converter is exposed in retriever.")
        raise AssertionError(f"Unexpected URL: {request.full_url}")

    def fake_site_open(request):
        if request.full_url == "https://charmm-gui.org/?doc=sign":
            return _FakeHTTPResponse(b"logged-in")
        if request.full_url == "https://charmm-gui.org/?doc=input/retriever&jobid=5804504324":
            return _FakeHTTPResponse(
                _make_retriever_html("5804504324", include_quick_bilayer=True, quick_bilayer_step="5").encode(
                    "utf-8"
                )
            )
        if request.full_url == "https://charmm-gui.org/?doc=user/jobids":
            return _FakeHTTPResponse(
                _make_jobids_html(
                    [
                        {
                            "jobid": "5804504324",
                            "project": "PDB Reader & Manipulator",
                            "module": "Quick Bilayer",
                            "step": "5",
                            "status": "Done",
                        }
                    ]
                ).encode("utf-8")
            )
        raise AssertionError(f"Unexpected site URL: {request.full_url}")

    monkeypatch.setattr(charmm_gui_module, "_urlopen", fake_urlopen)
    monkeypatch.setattr(charmm_gui_module, "_build_opener", lambda *args, **kwargs: _FakeSiteOpener(fake_site_open))

    result = backend.prepare_model(
        md,
        str(parameters_folder),
        workflow_mode="collect",
    )

    assert result.input_format == "remote_pending"
    assert result.metadata["state"] == "ffconverter_pending"
    assert "FF-Converter step" in result.metadata["reason"]
    assert result.metadata["retriever_steps"] == ["pdbreader:1", "pdbreader:2", "pdbreader:3", "membrane.quick:5"]


def test_charmm_gui_backend_collect_mode_returns_pending_when_archive_lacks_amber_inputs(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend(token="token", poll_interval_s=0.0, timeout_s=5.0)
    md = _build_openmm_md(tmp_path)
    parameters_folder = tmp_path / "parameters"
    model_root = parameters_folder / "modelA"
    model_root.mkdir(parents=True, exist_ok=True)
    cached_request = {
        "query": {
            "jobid": "12345",
            "margin": "15.0",
            "wdist": "22.5",
            "ion_conc": "0.15",
            "ion_type": "NaCl",
            "ppm": "false",
            "run_ffconverter": "true",
        },
        "form": {
            "upper": "POPC:POPE=7:3",
            "lower": "POPC:POPE=7:3",
        },
    }
    (model_root / "request.json").write_text(json.dumps(cached_request))
    (model_root / "submit_response.json").write_text(json.dumps({"jobid": "5804504324", "submitted": True}))

    download_bytes = _make_tarball(
        {
            "job/step1_pdbreader.pdb": "PDB",
            "job/step1_pdbreader.psf": "PSF",
        }
    )

    def fake_urlopen(request):
        if request.full_url.startswith("https://www.charmm-gui.org/api/check_status"):
            return _FakeHTTPResponse(json.dumps({"status": "done", "hasTarFile": True}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/download"):
            return _FakeHTTPResponse(download_bytes)
        raise AssertionError(f"Unexpected URL: {request.full_url}")

    monkeypatch.setattr(charmm_gui_module, "_urlopen", fake_urlopen)

    result = backend.prepare_model(
        md,
        str(parameters_folder),
        workflow_mode="collect",
    )

    assert result.input_format == "remote_pending"
    assert result.metadata["state"] == "download_incomplete"
    assert result.metadata["quick_bilayer_jobid"] == "5804504324"
    assert "AMBER inputs" in result.metadata["reason"]


def test_charmm_gui_backend_resolves_quick_bilayer_jobid_from_retriever_page(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend(email="user@example.org", password="secret", poll_interval_s=0.0, timeout_s=5.0)
    md = _build_openmm_md(tmp_path)
    parameters_folder = tmp_path / "parameters"

    download_bytes = _make_tarball(
        {
            "job/amber/step5_input.parm7": "PARM7",
            "job/amber/step5_input.rst7": "RST7",
        }
    )
    api_requests = []

    def fake_urlopen(request):
        api_requests.append(request.full_url)
        if request.full_url == "https://www.charmm-gui.org/api/login":
            return _FakeHTTPResponse(json.dumps({"token": "jwt-token"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/quick_bilayer"):
            return _FakeHTTPResponse(json.dumps({"jobid": "12345", "submitted": "true"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/check_status"):
            assert request.full_url.endswith("jobid=5804504324")
            return _FakeHTTPResponse(json.dumps({"status": "done"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/download"):
            assert request.full_url.endswith("jobid=5804504324")
            return _FakeHTTPResponse(download_bytes)
        raise AssertionError(f"Unexpected URL: {request.full_url}")

    def fake_site_open(request):
        if request.full_url == "https://charmm-gui.org/?doc=sign":
            return _FakeHTTPResponse(b"logged-in")
        if request.full_url == "https://charmm-gui.org/?doc=input/retriever&jobid=12345":
            return _FakeHTTPResponse(
                _make_retriever_html("12345", include_quick_bilayer=True, quick_bilayer_jobid="5804504324").encode(
                    "utf-8"
                )
            )
        if request.full_url == "https://charmm-gui.org/?doc=input/retriever&jobid=5804504324":
            return _FakeHTTPResponse(
                _make_retriever_html("5804504324", include_ffconverter=True, ffconverter_step="6").encode("utf-8")
            )
        raise AssertionError(f"Unexpected site URL: {request.full_url}")

    monkeypatch.setattr(charmm_gui_module, "_urlopen", fake_urlopen)
    monkeypatch.setattr(charmm_gui_module, "_build_opener", lambda *args, **kwargs: _FakeSiteOpener(fake_site_open))

    result = backend.prepare_model(
        md,
        str(parameters_folder),
        pdb_reader_jobid="12345",
        quick_bilayer={"membtype": "PMm", "margin": 20},
    )

    assert result.metadata["pdb_reader_jobid"] == "12345"
    assert result.metadata["quick_bilayer_jobid"] == "5804504324"
    assert (parameters_folder / "modelA" / "quick_bilayer_manifest.json").exists()
    assert any(url.endswith("jobid=5804504324") for url in api_requests)


def test_charmm_gui_backend_raises_when_quick_bilayer_never_activates_on_retriever_page(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend(
        email="user@example.org",
        password="secret",
        poll_interval_s=0.0,
        timeout_s=5.0,
        quick_bilayer_activation_poll_interval_s=0.0,
        quick_bilayer_activation_timeout_s=0.0,
    )
    md = _build_openmm_md(tmp_path)

    def fake_urlopen(request):
        if request.full_url == "https://www.charmm-gui.org/api/login":
            return _FakeHTTPResponse(json.dumps({"token": "jwt-token"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/quick_bilayer"):
            return _FakeHTTPResponse(json.dumps({"jobid": "12345", "submitted": "true"}).encode("utf-8"))
        raise AssertionError(f"Unexpected URL: {request.full_url}")

    def fake_site_open(request):
        if request.full_url == "https://charmm-gui.org/?doc=sign":
            return _FakeHTTPResponse(b"logged-in")
        if request.full_url == "https://charmm-gui.org/?doc=input/retriever&jobid=12345":
            return _FakeHTTPResponse(_make_retriever_html("12345", include_quick_bilayer=False).encode("utf-8"))
        if request.full_url == "https://charmm-gui.org/?doc=input/retriever&jobid=5804504324":
            return _FakeHTTPResponse(
                _make_retriever_html("5804504324", include_ffconverter=True, ffconverter_step="6").encode("utf-8")
            )
        if request.full_url == "https://charmm-gui.org/?doc=input/membrane.quick&jobid=12345&project=pdbreader":
            return _FakeHTTPResponse(b"<html><body>not ready</body></html>")
        if request.full_url == "https://charmm-gui.org/?doc=user/jobids":
            return _FakeHTTPResponse(_make_jobids_html([]).encode("utf-8"))
        raise AssertionError(f"Unexpected site URL: {request.full_url}")

    monkeypatch.setattr(charmm_gui_module, "_urlopen", fake_urlopen)
    monkeypatch.setattr(charmm_gui_module, "_build_opener", lambda *args, **kwargs: _FakeSiteOpener(fake_site_open))

    with pytest.raises(RuntimeError, match="did not activate"):
        backend.prepare_model(
            md,
            str(tmp_path / "parameters"),
            pdb_reader_jobid="12345",
            quick_bilayer={"membtype": "PMm", "margin": 20},
        )


def test_charmm_gui_backend_resolves_quick_bilayer_from_quick_page(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend(email="user@example.org", password="secret", poll_interval_s=0.0, timeout_s=5.0)
    md = _build_openmm_md(tmp_path)
    parameters_folder = tmp_path / "parameters"

    download_bytes = _make_tarball(
        {
            "job/amber/step5_input.parm7": "PARM7",
            "job/amber/step5_input.rst7": "RST7",
        }
    )

    def fake_urlopen(request):
        if request.full_url == "https://www.charmm-gui.org/api/login":
            return _FakeHTTPResponse(json.dumps({"token": "jwt-token"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/quick_bilayer"):
            return _FakeHTTPResponse(json.dumps({"jobid": "12345", "submitted": "true"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/check_status"):
            assert request.full_url.endswith("jobid=12345")
            return _FakeHTTPResponse(json.dumps({"status": "done"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/download"):
            assert request.full_url.endswith("jobid=12345")
            return _FakeHTTPResponse(download_bytes)
        raise AssertionError(f"Unexpected URL: {request.full_url}")

    def fake_site_open(request):
        if request.full_url == "https://charmm-gui.org/?doc=sign":
            return _FakeHTTPResponse(b"logged-in")
        if request.full_url == "https://charmm-gui.org/?doc=input/retriever&jobid=12345":
            return _FakeHTTPResponse(
                _make_retriever_html("12345", include_ffconverter=True, ffconverter_step="6").encode("utf-8")
            )
        if request.full_url == "https://charmm-gui.org/?doc=input/membrane.quick&jobid=12345&project=pdbreader":
            return _FakeHTTPResponse(_make_quick_page_html("12345", step="3").encode("utf-8"))
        raise AssertionError(f"Unexpected site URL: {request.full_url}")

    monkeypatch.setattr(charmm_gui_module, "_urlopen", fake_urlopen)
    monkeypatch.setattr(charmm_gui_module, "_build_opener", lambda *args, **kwargs: _FakeSiteOpener(fake_site_open))

    result = backend.prepare_model(
        md,
        str(parameters_folder),
        pdb_reader_jobid="12345",
        quick_bilayer={"membtype": "PMm", "margin": 20},
    )

    assert result.metadata["quick_bilayer_jobid"] == "12345"
    manifest = json.loads((parameters_folder / "modelA" / "quick_bilayer_manifest.json").read_text())
    assert manifest["jobid"] == "12345"
    assert manifest["step"] == "3"


def test_charmm_gui_backend_resolves_quick_bilayer_from_jobids_page_when_jobid_changes(tmp_path, monkeypatch):
    backend = CHARMMGUIBackend(email="user@example.org", password="secret", poll_interval_s=0.0, timeout_s=5.0)
    md = _build_openmm_md(tmp_path)
    parameters_folder = tmp_path / "parameters"

    download_bytes = _make_tarball(
        {
            "job/amber/step5_input.parm7": "PARM7",
            "job/amber/step5_input.rst7": "RST7",
        }
    )
    created_epoch = 1774815934

    def fake_urlopen(request):
        if request.full_url == "https://www.charmm-gui.org/api/login":
            return _FakeHTTPResponse(json.dumps({"token": "jwt-token"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/quick_bilayer"):
            return _FakeHTTPResponse(json.dumps({"jobid": "12345", "submitted": "true"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/check_status"):
            assert request.full_url.endswith("jobid=5804504324")
            return _FakeHTTPResponse(json.dumps({"status": "done"}).encode("utf-8"))
        if request.full_url.startswith("https://www.charmm-gui.org/api/download"):
            assert request.full_url.endswith("jobid=5804504324")
            return _FakeHTTPResponse(download_bytes)
        raise AssertionError(f"Unexpected URL: {request.full_url}")

    def fake_site_open(request):
        if request.full_url == "https://charmm-gui.org/?doc=sign":
            return _FakeHTTPResponse(b"logged-in")
        if request.full_url == "https://charmm-gui.org/?doc=input/retriever&jobid=12345":
            return _FakeHTTPResponse(_make_retriever_html("12345", include_quick_bilayer=False).encode("utf-8"))
        if request.full_url == "https://charmm-gui.org/?doc=input/membrane.quick&jobid=12345&project=pdbreader":
            return _FakeHTTPResponse(b"<html><body>not ready</body></html>")
        if request.full_url == "https://charmm-gui.org/?doc=user/jobids":
            return _FakeHTTPResponse(
                _make_jobids_html(
                    [
                        {
                            "jobid": "5804504324",
                            "project": "PDB Reader & Manipulator",
                            "module": "Quick Bilayer",
                            "step": "2",
                            "status": "Running",
                            "created_epoch": created_epoch,
                        }
                    ]
                ).encode("utf-8")
            )
        raise AssertionError(f"Unexpected site URL: {request.full_url}")

    monkeypatch.setattr(charmm_gui_module, "_urlopen", fake_urlopen)
    monkeypatch.setattr(charmm_gui_module, "_build_opener", lambda *args, **kwargs: _FakeSiteOpener(fake_site_open))
    monkeypatch.setattr(charmm_gui_module.time, "time", lambda: created_epoch)

    result = backend.prepare_model(
        md,
        str(parameters_folder),
        pdb_reader_jobid="12345",
        quick_bilayer={"membtype": "PMm", "margin": 20},
    )

    assert result.metadata["quick_bilayer_jobid"] == "5804504324"
    manifest = json.loads((parameters_folder / "modelA" / "quick_bilayer_manifest.json").read_text())
    assert manifest["jobid"] == "5804504324"
    assert manifest["step"] == "2"
