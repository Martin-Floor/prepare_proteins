"""Unit tests for the preparation-only AlphaFold 3 input API.

``sequenceModels.writeAlphaFold3Inputs`` exists so that a staged workflow can materialize
AlphaFold 3 JSON inputs without generating or launching any scheduler command, and
``setUpAlphaFold3`` reuses the same builder and writer. These tests pin both halves of that
contract: what the preparation-only mode writes, and that the command-generating method still
behaves exactly as before.
"""

import json
import os

import prepare_proteins
import pytest
from prepare_proteins._prepare_sequences import (
    alphafold3_sanitised_name,
    build_alphafold3_payload,
    validate_alphafold3_payload,
)

SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"
OTHER_SEQUENCE = "MKVLAAGGTTWY"


def a3m(sequence, homologs=2):
    lines = [f">query\n{sequence}"]
    lines += [f">homolog_{index}\n{sequence}" for index in range(1, homologs + 1)]
    return "\n".join(lines) + "\n"


def test_write_inputs_materializes_json_without_any_scheduler_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = prepare_proteins.sequenceModels({"modelA": SEQUENCE, "modelB": OTHER_SEQUENCE})

    records = models.writeAlphaFold3Inputs(str(tmp_path / "af3"), model_seeds=[7, 8])

    assert [record["model"] for record in records] == ["modelA", "modelB"]
    for record in records:
        payload = json.loads(open(record["path"]).read())
        assert payload == record["payload"]
        assert payload["dialect"] == "alphafold3" and payload["version"] == 1
        assert payload["modelSeeds"] == [7, 8]
        assert os.path.isdir(record["output_dir"])
    # Nothing the preparation-only API returns or writes may carry a submission command.
    blob = json.dumps(records, sort_keys=True, default=str)
    assert "sbatch" not in blob
    assert "bsc_alphafold" not in blob
    for record in records:
        assert "sbatch" not in open(record["path"]).read()


def test_write_inputs_round_trips_msa_template_and_modification_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = prepare_proteins.sequenceModels({"modelA": {"A": SEQUENCE, "B": SEQUENCE}})

    records = models.writeAlphaFold3Inputs(
        str(tmp_path / "af3"),
        model_seeds=1,
        ligands={"modelA": [{"id": "LS", "smiles": "OCC(O)CO"},
                            {"id": "LM", "ccdCodes": ["MG"]}]},
        unpaired_msa=a3m(SEQUENCE),
        paired_msa="",
        templates=[],
        modifications={"modelA": {"A": [{"ptmType": "CCD_SEP", "ptmPosition": 16}],
                                  "B": []}},
        require_msa=True,
    )

    payload = records[0]["payload"]
    proteins = [entry["protein"] for entry in payload["sequences"] if "protein" in entry]
    ligands = [entry["ligand"] for entry in payload["sequences"] if "ligand" in entry]
    assert [protein["id"] for protein in proteins] == ["A", "B"]
    assert all(protein["unpairedMsa"] == a3m(SEQUENCE) for protein in proteins)
    assert all(protein["pairedMsa"] == "" for protein in proteins)
    assert all(protein["templates"] == [] for protein in proteins)
    assert proteins[0]["modifications"] == [{"ptmType": "CCD_SEP", "ptmPosition": 16}]
    assert proteins[1]["modifications"] == []
    # Ordered ligand entries keep the caller's chain layout, CCD and SMILES side by side.
    assert ligands == [{"id": "LS", "smiles": "OCC(O)CO"}, {"id": "LM", "ccdCodes": ["MG"]}]
    assert json.loads(open(records[0]["path"]).read()) == payload
    assert not validate_alphafold3_payload(payload, require_msa=True)


def test_write_inputs_selects_msa_per_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = prepare_proteins.sequenceModels({"modelA": SEQUENCE, "modelB": OTHER_SEQUENCE})

    records = models.writeAlphaFold3Inputs(
        str(tmp_path / "af3"),
        unpaired_msa={"modelA": a3m(SEQUENCE), "modelB": a3m(OTHER_SEQUENCE)},
        paired_msa={"default": ""},
        templates={"*": []},
    )

    by_model = {record["model"]: record["payload"] for record in records}
    assert by_model["modelA"]["sequences"][0]["protein"]["unpairedMsa"] == a3m(SEQUENCE)
    assert by_model["modelB"]["sequences"][0]["protein"]["unpairedMsa"] == a3m(OTHER_SEQUENCE)
    assert all(payload["sequences"][0]["protein"]["templates"] == []
               for payload in by_model.values())


def test_write_inputs_flat_layout_uses_payload_and_file_name_overrides(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = prepare_proteins.sequenceModels({"CTRL-01": SEQUENCE})

    records = models.writeAlphaFold3Inputs(
        str(tmp_path / "inputs"),
        layout="flat",
        payload_names={"CTRL-01": "ctrl-01_msa"},
        file_names={"CTRL-01": "CTRL-01-msa"},
        json_sort_keys=True,
        require_sanitised_name=True,
    )

    path = tmp_path / "inputs" / "CTRL-01-msa.json"
    assert records[0]["path"] == str(path)
    assert "model_dir" not in records[0]
    text = path.read_text()
    assert json.loads(text)["name"] == "ctrl-01_msa"
    assert text == json.dumps(records[0]["payload"], indent=2, sort_keys=True) + "\n"
    assert alphafold3_sanitised_name("ctrl-01_msa") == "ctrl-01_msa"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"unpaired_msa": {"modelZ": "x"}}, "unknown model names"),
        ({"unpaired_msa": 3}, "must be a string"),
        ({"templates": "none"}, "templates must be a list"),
        ({"templates": [{"mmcif": "data_X", "mmcifPath": "/x.cif", "queryIndices": [0],
                         "templateIndices": [0]}]}, "exactly one of mmcif"),
        ({"templates": [{"mmcif": "data_X", "queryIndices": [0]}]}, "missing templateIndices"),
        ({"modifications": [{"ptmType": "CCD_SEP", "ptmPosition": 999}]}, "outside the 1-based"),
        ({"modifications": [{"ptmType": "CCD_SEP"}]}, "missing"),
        ({"modifications": [{"ptmType": "CCD_SEP", "ptmPosition": 1, "chain": "A"}]},
         "unsupported keys"),
        ({"ligands": {"modelA": [{"id": "LS", "smiles": "CCO", "ccdCodes": ["MG"]}]}},
         "exactly one of ccdCodes or smiles"),
    ],
)
def test_write_inputs_refuses_invalid_field_specifications(tmp_path, monkeypatch, kwargs,
                                                           message):
    monkeypatch.chdir(tmp_path)
    models = prepare_proteins.sequenceModels({"modelA": SEQUENCE})

    with pytest.raises((ValueError, TypeError), match=message):
        models.writeAlphaFold3Inputs(str(tmp_path / "af3"), **kwargs)


def test_write_inputs_refuses_conflicting_per_chain_values_in_one_entry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = prepare_proteins.sequenceModels({"modelA": {("A", "B"): SEQUENCE}})

    with pytest.raises(ValueError, match="differs between the chains"):
        models.writeAlphaFold3Inputs(
            str(tmp_path / "af3"),
            unpaired_msa={"modelA": {"A": a3m(SEQUENCE), "B": a3m(OTHER_SEQUENCE)}},
        )


def test_write_inputs_validation_rejects_an_unsanitised_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = prepare_proteins.sequenceModels({"Model A": SEQUENCE})

    with pytest.raises(ValueError, match="sanitised"):
        models.writeAlphaFold3Inputs(str(tmp_path / "af3"), require_sanitised_name=True)


def test_build_payloads_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = prepare_proteins.sequenceModels({"modelA": SEQUENCE})
    target = tmp_path / "af3"

    specs = models.buildAlphaFold3Payloads(model_seeds=[3])

    assert not target.exists()
    assert specs[0]["payload"]["modelSeeds"] == [3]
    assert specs[0]["payload_name"] == "modelA"


def test_set_up_alphafold3_writes_the_same_bytes_as_the_preparation_only_api(tmp_path,
                                                                            monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = prepare_proteins.sequenceModels({"modelA": {"A": SEQUENCE, "B": SEQUENCE}})
    shared = dict(model_seeds=[11], ligands={"modelA": [{"id": "LM", "ccdCodes": ["MG"]}]},
                  unpaired_msa=a3m(SEQUENCE), paired_msa="", templates=[])

    commands = models.setUpAlphaFold3(str(tmp_path / "commands"), **shared)
    models.writeAlphaFold3Inputs(str(tmp_path / "prepared"), **shared)

    from_commands = (tmp_path / "commands" / "modelA" / "input" / "modelA.json").read_text()
    from_writer = (tmp_path / "prepared" / "modelA" / "input" / "modelA.json").read_text()
    assert from_commands == from_writer
    # The command-generating method still generates commands; the writer never does.
    assert len(commands) == 1 and "sbatch runner input" in commands[0]


def test_validate_alphafold3_payload_reports_the_reason_for_each_defect():
    payload = build_alphafold3_payload(
        "job", [{"protein": {"id": "A", "sequence": SEQUENCE, "unpairedMsa": a3m(SEQUENCE),
                             "pairedMsa": "", "templates": []}},
                {"ligand": {"id": "LM", "ccdCodes": ["MG"]}}], [1])
    assert not validate_alphafold3_payload(payload, require_msa=True,
                                           require_sanitised_name=True)

    def failures(mutate, **kwargs):
        broken = json.loads(json.dumps(payload))
        mutate(broken)
        return " ".join(validate_alphafold3_payload(broken, **kwargs))

    assert "dialect" in failures(lambda p: p.update({"dialect": "alphafoldserver"}))
    assert "version" in failures(lambda p: p.update({"version": 2}))
    assert "unexpected" in failures(lambda p: p.update({"temperature": 1.0}))
    assert "modelSeeds" in failures(lambda p: p.update({"modelSeeds": []}))
    assert "sanitised" in failures(lambda p: p.update({"name": "Job One"}),
                                   require_sanitised_name=True)
    assert "duplicate" in failures(lambda p: p["sequences"].append(dict(p["sequences"][0])))
    assert "exactly one of ccdCodes" in failures(
        lambda p: p["sequences"][1]["ligand"].update({"smiles": "CCO"}))
    assert "SMILES is not parsable" in failures(
        lambda p: p["sequences"][1]["ligand"].update({"ccdCodes": None,
                                                      "smiles": "not a smiles ((("}))
    assert "non-empty unpairedMsa" in failures(
        lambda p: p["sequences"][0]["protein"].update({"unpairedMsa": "  "}), require_msa=True)
    assert "pairedMsa that is present" in failures(
        lambda p: p["sequences"][0]["protein"].update({"pairedMsa": None}), require_msa=True)
    assert "empty list" in failures(
        lambda p: p["sequences"][0]["protein"].pop("templates"), require_msa=True)
    assert "unsupported keys" in failures(
        lambda p: p["sequences"][0]["protein"].update({"msaPath": "/tmp/x.a3m"}))


def test_build_alphafold3_payload_refuses_both_user_ccd_forms():
    with pytest.raises(ValueError, match="cannot define both"):
        build_alphafold3_payload("job", [{"protein": {"id": "A", "sequence": SEQUENCE}}], [1],
                                 user_ccd="data_X\n#\n", user_ccd_path="/tmp/x.cif")
