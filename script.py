import subprocess
import sys

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(error.decode('utf-8'))
    else:
        print(output.decode('utf-8'))

def uninstall_packages(packages):
    for package in packages:
        print(f"Uninstalling {package}...")
        run_command(f"{sys.executable} -m pip uninstall -y {package}")

def install_packages(packages):
    for package in packages:
        print(f"Installing {package}...")
        run_command(f"{sys.executable} -m pip install {package}")

if __name__ == "__main__":
    packages = [
        "airsim",
        "msgpack-rpc-python",
        "numpy",
        "opencv-python",
        "tornado==4.5.3"  # Specific version that works well with AirSim
    ]

    print("Uninstalling packages...")
    uninstall_packages(packages)

    print("\nInstalling packages...")
    install_packages(packages)

    print("\nUpgrading pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")

    print("\nInstallation complete. Please restart your Python environment.")