from tests import (
    test_er_encoder,
    test_format_validator,
    test_generator,
    test_helper,
    test_mb_encoder,
    test_runner,
    test_solver,
    test_utils
)

def get_unittests(suite):
    suite.addTest(unittest.makeSuite(test_format_validator.TestFormatValidator))
    suite.addTest(unittest.makeSuite(test_generator.TestGenerator))
    suite.addTest(unittest.makeSuite(test_helper.TestHelper))
    suite.addTest(unittest.makeSuite(test_utils.TestUtils))

def get_encoder_tests(suite):
    suite.addTest(unittest.makeSuite(test_er_encoder.TestEREncoder))
    suite.addTest(unittest.makeSuite(test_mb_encoder.TestMBEncoder))
    suite.addTest(unittest.makeSuite(test_solver.TestSolver))
    suite.addTest(unittest.makeSuite(test_runner.TestRunner))

def main():
    suite = unittest.TestSuite()
    get_unittests(suite)
    get_server_tests(suite)

    output = unittest.TextTestRunner(verbosity=2).run(suite)
    if output.errors or output.failures:
        print("Failing Tests")


if __name__ == "__main__":
    main()