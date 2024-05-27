"""Write data files properties."""

from homecredit import utils

dfs_props = {
    "base": {
        "paths": ["base"],
        "features_agg": []
    },
    "person1": {
        "paths": ["person_1"],
        "features_agg": ["formulas"]
    },
    "static0": {
        "paths": ["static_0_*"],
        "features_agg": []
    },
    "staticcb0": {
        "paths": ["static_cb_0"],
        "features_agg": []
    },

    "other1": {
        "paths": ["other_1"],
        "features_agg": ["formulas"]
    },
    "deposit1": {
        "paths": ["deposit_1"],
        "features_agg": ["formulas"]
    },
    "debitcard1": {
        "paths": ["debitcard_1"],
        "features_agg": ["formulas"]
    },
    "applprev1": {
        "paths": ["applprev_1_*"],
        "features_agg": ["formulas"]
    },
    "applprev2": {
        "paths": ["applprev_2"],
        "features_agg": ["formulas"]
    },
    "creditbureau1": {
        "paths": ["credit_bureau_a_1_*"],#, "credit_bureau_b_1"],
        "features_agg": ["formulas"]
    },
    "creditbureau2": {
        "paths": ["credit_bureau_a_2_*"],#, "credit_bureau_b_2"],
        "features_agg": ["formulas"]
    },
    "taxregistrya1": {
        "paths": ["tax_registry_a_1"],
        "features_agg": ["formulas"]
    },
    "taxregistryb1": {
        "paths": ["tax_registry_b_1"],
        "features_agg": ["formulas"]
    },
    "taxregistryc1": {
        "paths": ["tax_registry_c_1"],
        "features_agg": ["formulas"]
    },
    "person2": {
        "paths": ["person_2"],
        "features_agg": ["formulas"]
    },
}

utils.write_data_props(dfs_props, version="3")