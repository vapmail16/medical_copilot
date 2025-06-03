import os
os.environ["OPENAI_API_KEY"] = "test_key"

import unittest
import coverage
import sys
from tests.test_medical_workflow import TestMedicalWorkflow

def run_tests():
    # Start coverage measurement
    cov = coverage.Coverage()
    cov.start()
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestMedicalWorkflow))
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Stop coverage measurement
    cov.stop()
    cov.save()
    
    # Print coverage report
    print("\nCoverage Report:")
    cov.report()
    
    # Generate HTML coverage report
    cov.html_report(directory='coverage_report')
    
    # Return test result
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 