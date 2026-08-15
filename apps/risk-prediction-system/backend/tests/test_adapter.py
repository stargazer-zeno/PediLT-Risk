import json

from app.adapters.standard_adapter import StandardDataAdapter, StandardDataAdapterError

adapter = StandardDataAdapter()


def test_single_json_object():
    obj = {"id": "p1", "基础信息": {"患儿性别": "男"}}
    out = adapter.from_json_obj(obj)
    assert len(out) == 1
    assert out[0]["id"] == "p1"
    # missing sections are filled in
    assert out[0]["时序检验指标 (纯数值序列)"] == []


def test_json_list_and_wrapper():
    assert len(adapter.from_json_obj([{"id": "a"}, {"id": "b"}])) == 2
    assert len(adapter.from_json_obj({"patients": [{"id": "a"}]})) == 1


def test_jsonl_bytes():
    text = '{"id": "a"}\n\n{"id": "b"}\n'
    out = adapter.from_bytes(text.encode("utf-8"), "x.jsonl")
    assert [p["id"] for p in out] == ["a", "b"]


def test_csv_flat_columns():
    csv_text = (
        "id,患儿性别,手术年龄,lab:ALB,med:环孢素,fu:术后天数\n"
        "p1,男,0.79,21.0;22.0,20.0,1;30\n"
    )
    out = adapter.from_bytes(csv_text.encode("utf-8"), "x.csv")
    assert len(out) == 1
    p = out[0]
    assert p["id"] == "p1"
    assert p["基础信息"]["患儿性别"] == "男"
    assert "ALB: 21.0, 22.0。" in p["时序检验指标 (纯数值序列)"]
    assert "环孢素: 20.0。" in p["时序用药记录 (纯数值序列)"]
    assert "术后天数: 1, 30。" in p["时序随访基础信息"]


def test_csv_patient_json_column():
    patient = {"id": "p9", "基础信息": {"患儿性别": "女"}}
    csv_text = "patient_json\n" + '"' + json.dumps(patient).replace('"', '""') + '"\n'
    out = adapter.from_bytes(csv_text.encode("utf-8"), "x.csv")
    assert out[0]["id"] == "p9"
    assert out[0]["基础信息"]["患儿性别"] == "女"


def test_unsupported_extension():
    try:
        adapter.from_bytes(b"x", "x.txt")
        assert False, "should raise"
    except StandardDataAdapterError:
        pass
