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

UNIQUE_SET_IDS = ["23588_T922WCPI"]
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