# How to Fix the "Openscan Meanwhile" App After a Failed Update

If the "Openscan Meanwhile" app is not working correctly or if the system keeps rebooting after a failed update, you can restore the application to a previous working state. This process involves switching between different **branches** of the app (such as **stable**, **beta**, or **meanwhile**).

### Prerequisites
- You need to be connected to the Raspberry Pi via SSH.  
  See the [how to access remote ssh](./Remote_ssh.md) for instructions on how to connect via SSH on Windows, Linux, or macOS.

---

## Step 1: Check the Current Branch

The app runs in one of three different **branches**: `stable`, `beta`, or `meanwhile`. To check which branch is currently active:

1. Once connected via SSH, run the following command to see the current branch:
   ```bash
   cat /home/pi/OpenScan/settings/openscan_branch
   ```
The terminal will display one of the following values:

* stable
* beta
* meanwhile

## Step 2: Change the Branch (if needed)
If you need to switch to a different branch (e.g., from beta to stable), you can manually update the configuration file:

Open the branch configuration file with this command:

```bash
nano /home/pi/OpenScan/settings/openscan_branch
```
Use your keyboard to modify the value to one of the following branches:


* stable
* beta
* meanwhile

Once you've updated the file, press Ctrl + X to exit, then Y to confirm the changes, and Enter to save.

## Step 3: Run the Emergency Update Script
After changing the branch (if necessary), run the emergency update script to apply the changes and restore the app to the working state:

1. Run the following command:

```bash
sudo /usr/local/bin/emergency_update
```
2. Wait for the script to complete. This will restore the application to the state of the selected branch.

## Additional Notes
This process is useful if the system starts rebooting after a failed update.
Switching branches helps you restore the app to a previous working version.
Ensure you are connected via SSH before proceeding with these steps (see [how to access to remote ssh](./Remote_ssh.md) for help).