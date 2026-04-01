from vote_simulation.simulation.simulation import simulation_from_file, simulation_series

if __name__ == "__main__":
    rules_codes = [
        "RV",
        "MJ",
        "AP_T",
        "AP_k",
        "BUCK_I",
        "BORD",
        "STAR",
        "BLAC",
        "SCHU",
        "COPE",
        "MMAX",
        "NANS",
        "COOM",
        "HARE",
        "PLU2",
        "PLU1",
    ]

    step_result = simulation_from_file("data/gen/UNI_v1001_c14/iter_0001.parquet", rules_codes)
    # print(step_result)
    step_result.plot_distance_matrix()

    series = simulation_series("data/gen/IC_v1001_c14", rules_codes)
    series.plot_mean_distance_matrix()

#    series_result =

"""rules_codes = [
    "AP_T",
    "AP_K",
    "BALD",
    "BLAC",
    "BORD",
    "CAIR",
    "CSUM",
    "CVIR",
    "BUCK_I",
    "BUCK_R",
    "COOM",
    "COPE",
    "DODG_C",
    "DODG_S",
    "EXHB",
    "HARE",
    "IRV",
    "IRVA",
    "IRVD",
    "ICRV",
    "KIMR",
    "MJ",
    "MMAX",
    "NANS",
    "PLU1",
    "PLU2",
    "RPAR",
    "RV",
    "SCHU",
    "SIRV",
    "SPCY",
    "STAR",
    "TIDE",
    "VETO",
    "WOOD",
    "YOUN"]
    """
