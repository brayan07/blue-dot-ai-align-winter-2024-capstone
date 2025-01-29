from kedro.pipeline import Pipeline, node
from .nodes import run_naive_for_unique_set, run_expert_for_unique_set, run_consultancy_for_unique_set, \
    run_unstructured_debate_for_unique_set, run_structured_debate_for_unique_set


def _create_naive_node_for_unique_set(set_unique_id: str):
    kind = "naive"
    return node(
        func=run_naive_for_unique_set,
        inputs=[f"quality_filtered_train_{set_unique_id}","params:llm_config"],
        outputs=f"unique_set_{kind}_results_{set_unique_id}",
        name=f"run_naive_for_unique_set_{set_unique_id}",
    ).tag("naive")

def _create_expert_node_for_unique_set(set_unique_id: str):
    kind = "expert"
    return node(
        func=run_expert_for_unique_set,
        inputs=[f"quality_filtered_train_{set_unique_id}","params:llm_config"],
        outputs=f"unique_set_{kind}_results_{set_unique_id}",
        name=f"run_expert_for_unique_set_{set_unique_id}",
    ).tag("expert")

def _create_consultancy_node_for_unique_set(set_unique_id: str):
    kind = "consultancy"
    return node(
        func=run_consultancy_for_unique_set,
        inputs=[f"quality_filtered_train_{set_unique_id}","params:llm_config"],
        outputs=f"unique_set_{kind}_results_{set_unique_id}",
        name=f"run_consultancy_for_unique_set_{set_unique_id}",
    ).tag("consultancy")

def _create_unstructured_debate_node_for_unique_set(set_unique_id: str):
    kind = "unstructured_debate"
    return node(
        func=run_unstructured_debate_for_unique_set,
        inputs=[f"quality_filtered_train_{set_unique_id}","params:llm_config"],
        outputs=f"unique_set_{kind}_results_{set_unique_id}",
        name=f"run_unstructured_debate_for_unique_set_{set_unique_id}",
    ).tag("unstructured_debate")

def _run_structured_debate_for_unique_set(set_unique_id: str):
    kind = "structured_debate"
    return node(
        func=run_structured_debate_for_unique_set,
        inputs=[f"quality_filtered_train_{set_unique_id}","params:llm_config"],
        outputs=f"unique_set_{kind}_results_{set_unique_id}",
        name=f"run_structured_debate_for_unique_set_{set_unique_id}",
    ).tag("structured_debate")

UNIQUE_SET_IDS = [
    '52995_I3M5VUMM',
    '63477_65UJ979R',
    '62349_N0MX51FA',
    '63109_XYACUEX2',
    '52995_X9XZD7EN',
    '63477_O9LLFQB4',
    '62349_5BTGFFCO',
    '62324_PSKZR17W',
    '63919_NSBYQCZW',
    '52844_ZW8X7FHT',
    '61405_DPAPHI73',
    '61139_QIEA2CJB',
    '63631_6I0C1TOX',
    '52326_CTLJIFOK',
    '63304_C7MZHCZM',
    '63640_WZ2MF15P',
    '63631_J5XWKN5C',
    '63304_NMANYPP2',
    '61097_S3UO0IYW',
    '62198_H1IWTV7E',
    '61097_L4LGF3WL',
    '62619_MI3FWOJ8',
    '61228_3A5O28VM',
    '61242_4XEEXVB0',
    '60507_5EHIDPFU',
    '63875_TA3WI7DW',
    '61052_GL60ZD9B',
    '63875_B507K45X',
    '61081_9X59TFEH',
    '61204_7K3R71T6'
]
def create_pipeline(**kwargs):
    nodes = [
       _create_naive_node_for_unique_set(set_unique_id) for set_unique_id in UNIQUE_SET_IDS
    ]
    nodes.extend([
        _create_expert_node_for_unique_set(set_unique_id) for set_unique_id in UNIQUE_SET_IDS
    ])
    nodes.extend([
        _create_consultancy_node_for_unique_set(set_unique_id) for set_unique_id in UNIQUE_SET_IDS
    ])
    nodes.extend([
        _create_unstructured_debate_node_for_unique_set(set_unique_id) for set_unique_id in UNIQUE_SET_IDS
    ])
    nodes.extend([
        _run_structured_debate_for_unique_set(set_unique_id) for set_unique_id in UNIQUE_SET_IDS
    ])
    return Pipeline(
        nodes=nodes
    )