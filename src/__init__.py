from .utils import get_file_path


# JSON File Paths
default_jfp = get_file_path("testfiles", "rgp_hc.json")

pos_jfp = get_file_path("testfiles", "rgp_test_pos.json")
neg_jfp = get_file_path("testfiles", "rgp_test_neg.json")
small_jfp = get_file_path("testfiles", "rgp_test_small.json")

multi_jfp = get_file_path("testfiles", "rgp_hc_multiple.json")
false_fmt_jfp = get_file_path("testfiles", "rgp_false_fmt.json")

test_path_neg = get_file_path("testfiles", "rgp_gen_0.json")
test_path_pos = get_file_path("testfiles", "rgp_gen_1.json")

experiment_config_path = get_file_path("configfiles", "experiment_config.json")