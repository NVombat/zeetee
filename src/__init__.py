from .utils import get_file_path


data_dir = "data"
test_sub_dir = "testfiles"
config_sub_dir = "configfiles"

assets_dir = "assets"
files_sub_dir = "files"
results_sub_dir = "results"

# JSON File Paths
default_jfp = get_file_path(data_dir, test_sub_dir, "rgp_hc.json")

pos_jfp = get_file_path(data_dir, test_sub_dir, "rgp_test_pos.json")
neg_jfp = get_file_path(data_dir, test_sub_dir, "rgp_test_neg.json")
small_jfp = get_file_path(data_dir, test_sub_dir, "rgp_test_small.json")

multi_jfp = get_file_path(data_dir, test_sub_dir, "rgp_hc_multiple.json")
false_fmt_jfp = get_file_path(data_dir, test_sub_dir, "rgp_false_fmt.json")

test_path_neg = get_file_path(data_dir, test_sub_dir, "rgp_gen_test_0.json")
test_path_pos = get_file_path(data_dir, test_sub_dir, "rgp_gen_test_1.json")
test_path_ran = get_file_path(data_dir, test_sub_dir, "rgp_gen_test_2.json")

exp_path = get_file_path(assets_dir, files_sub_dir, "rgp_gen_exp.json")

experiment_config_path = get_file_path(data_dir, config_sub_dir, "experiment_config.json")