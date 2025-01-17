# run_tests

import subprocess
import os

def list_tests():
    """Lists all available tests in the `tests` directory"""
    test_files = [f for f in os.listdir('tests') if f.startswith('test_') and f.endswith('.py')]
    return test_files

def run_all_tests():
    """Runs all tests"""
    subprocess.run(['pytest', 'tests'], check=True)

def run_selected_test(test_name):
    """Runs a specific test"""
    subprocess.run(['pytest', f'tests/{test_name}'], check=True)

def main():
    print("Select an option:")
    print("1. Run all tests")
    print("2. Run a specific test")

    option = input("Enter your choice (1/2): ")

    if option == '1':
        run_all_tests()
    elif option == '2':
        print("Available tests:")
        tests = list_tests()
        for i, test in enumerate(tests, 1):
            print(f"{i}. {test}")
        
        choice = int(input(f"Enter the number of the test you want to run (1-{len(tests)}): "))
        
        if 1 <= choice <= len(tests):
            selected_test = tests[choice - 1]
            run_selected_test(selected_test)
        else:
            print("Invalid choice.")
    else:
        print("Invalid option.")

if __name__ == "__main__":
    main()
