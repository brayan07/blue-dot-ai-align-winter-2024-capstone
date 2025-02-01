from kedro.pipeline import Pipeline, node
from .nodes import (
    run_naive_for_unique_set,
    run_expert_for_unique_set,
    run_consultancy_for_unique_set,
    run_unstructured_debate_for_unique_set,
    run_structured_debate_for_unique_set,
)


def _create_naive_node_for_unique_set(set_unique_id: str):
    kind = "naive"
    return node(
        func=run_naive_for_unique_set,
        inputs=[f"quality_filtered_train_{set_unique_id}", "params:llm_config"],
        outputs=f"unique_set_{kind}_results_{set_unique_id}",
        name=f"run_naive_for_unique_set_{set_unique_id}",
    ).tag("naive")


def _create_expert_node_for_unique_set(set_unique_id: str):
    kind = "expert"
    return node(
        func=run_expert_for_unique_set,
        inputs=[f"quality_filtered_train_{set_unique_id}", "params:llm_config"],
        outputs=f"unique_set_{kind}_results_{set_unique_id}",
        name=f"run_expert_for_unique_set_{set_unique_id}",
    ).tag("expert")


def _create_consultancy_node_for_unique_set(set_unique_id: str):
    kind = "consultancy"
    return node(
        func=run_consultancy_for_unique_set,
        inputs=[f"quality_filtered_train_{set_unique_id}", "params:llm_config"],
        outputs=f"unique_set_{kind}_results_{set_unique_id}",
        name=f"run_consultancy_for_unique_set_{set_unique_id}",
    ).tag("consultancy")


def _create_unstructured_debate_node_for_unique_set(set_unique_id: str):
    kind = "unstructured_debate"
    return node(
        func=run_unstructured_debate_for_unique_set,
        inputs=[f"quality_filtered_train_{set_unique_id}", "params:llm_config"],
        outputs=f"unique_set_{kind}_results_{set_unique_id}",
        name=f"run_unstructured_debate_for_unique_set_{set_unique_id}",
    ).tag("unstructured_debate")


def _run_structured_debate_for_unique_set(set_unique_id: str):
    kind = "structured_debate"
    return node(
        func=run_structured_debate_for_unique_set,
        inputs=[f"quality_filtered_train_{set_unique_id}", "params:llm_config"],
        outputs=f"unique_set_{kind}_results_{set_unique_id}",
        name=f"run_structured_debate_for_unique_set_{set_unique_id}",
    ).tag("structured_debate")


# All unique set ids in the training set
UNIQUE_SET_IDS = [
    # '23588_T922WCPI',
    # '23592_UIJQGZDK',
    "23791_S6420G0B",
    "23942_IGIFD97I",
    "24161_8CTVPP0F",
    "24161_INDOEF2N",
    "24192_35VUFWCR",
    "24278_K2R6V1ZI",
    "24278_LON5P1ZP",
    "24958_5FOW0VR7",
    "24966_5VVG9W0A",
    "24977_63MMCMYX",
    "25086_J5I8Y7L0",
    "25086_TN2QYF3S",
    "25627_61MWU4OY",
    "25627_L0AQPROP",
    "26066_9JVKF36B",
    "26569_CEKEK4QL",
    "26569_ZA7RADIT",
    "26741_OUX1V2UX",
    # ------
    # '26741_OVYFBIST',
    # '26843_JEQCNBC3',
    # '27110_HKV3Z17H',
    # '27665_07JE5ME7',
    # '27665_6ZYKASVR',
    # '29196_HBX60GQ0',
    # '31355_7INGZJ49',
    # '31355_N2HLEA08',
    # '31357_T9I0O70O',
    # '32836_VRNCK2U5',
    # '47841_OGLZAMCM',
    # '47841_Z78BHWVX',
    # '49165_DHRS39DU',
    # '50103_C2KT9HTQ',
    # '50103_MXGA7R12',
    # '50766_FZPG0W49',
    # '50869_296A3IP6',
    # '50969_Z49UU7OO',
    # '50988_9Z74IVOZ',
    # '51046_ILVOCQ22',
    # '51092_R7259EYH',
    # '51129_QKRGX6TG',
    # '51170_N4I8DROP',
    # '51170_SHLS4CGY',
    # '51203_DAOXSI8A',
    # '51249_8LFO3G16',
    # '51286_6W2Z4K2V',
    # '51296_JQIUN4AC',
    # '51330_YVTDCBLI',
    # '51337_QQIKBEZ3',
    # '51361_Q2HT9US4',
    # '51361_RYX1GAGT',
    # '51433_04A0V67E',
    # '51494_06CS6DYL',
    # '51494_CFQPFY28',
    # '51597_L4VT5NLR',
    # '51609_0LV2K87T',
    # '51609_CP0LRIY3',
    # '51656_1AL2TW0L',
    # '51657_4L2YAFRQ',
    # '52326_CTLJIFOK',
    # '52844_ZW8X7FHT',
    # '52995_I3M5VUMM',
    # '52995_X9XZD7EN',
    # '53016_JX0ZBJW2',
    # '60507_5EHIDPFU',
    # '60515_1N5NTY6S',
    # '60713_KMWD720A',
    # '60713_TSA8I2K7',
    # '60995_57L7VNGG',
    # '60995_B2JLL3Y9',
    # '61052_GL60ZD9B',
    # '61081_9X59TFEH',
    # '61097_L4LGF3WL',
    # '61097_S3UO0IYW',
    # '61139_QIEA2CJB',
    # '61204_7K3R71T6',
    # '61213_8GQZLKML',
    # '61228_3A5O28VM',
    # '61242_4XEEXVB0',
    # '61405_DPAPHI73',
    # '61481_LZNKW9Z1',
    # '62198_H1IWTV7E',
    # '62324_PSKZR17W',
    # '62349_5BTGFFCO',
    # '62349_N0MX51FA',
    # '62619_MI3FWOJ8',
    # '63109_XYACUEX2',
    # '63304_C7MZHCZM',
    # '63304_NMANYPP2',
    # '63477_65UJ979R',
    # '63477_O9LLFQB4',
    # '63631_6I0C1TOX',
    # '63631_J5XWKN5C',
    # '63640_WZ2MF15P',
    # '63875_B507K45X',
    # '63875_TA3WI7DW',
    # '63890_OZY8SIE2',
    # '63919_NSBYQCZW'
]


def create_pipeline(**kwargs):
    nodes = [
        _create_naive_node_for_unique_set(set_unique_id)
        for set_unique_id in UNIQUE_SET_IDS
    ]
    nodes.extend(
        [
            _create_expert_node_for_unique_set(set_unique_id)
            for set_unique_id in UNIQUE_SET_IDS
        ]
    )
    nodes.extend(
        [
            _create_consultancy_node_for_unique_set(set_unique_id)
            for set_unique_id in UNIQUE_SET_IDS
        ]
    )
    nodes.extend(
        [
            _create_unstructured_debate_node_for_unique_set(set_unique_id)
            for set_unique_id in UNIQUE_SET_IDS
        ]
    )
    nodes.extend(
        [
            _run_structured_debate_for_unique_set(set_unique_id)
            for set_unique_id in UNIQUE_SET_IDS
        ]
    )
    return Pipeline(nodes=nodes)
