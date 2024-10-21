# How to Access Raspberry Pi via SSH on Different Platforms

You can access the Raspberry Pi (`openscan.local` on port 22) via SSH using the following instructions, depending on your operating system.

## General Information
- **Raspberry Pi hostname**: `openscan.local`
- **SSH port**: `22`
- **Username & Password**: The default is usually `pi` (username) and `raspberry` (password), unless changed.

---

## 1. Windows (Using PuTTY)
To access the Raspberry Pi via SSH on Windows, you will use **PuTTY**, a popular SSH client.

### Steps:
1. **Download PuTTY**:  
   - Go to [the PuTTY download page](https://www.putty.org/) and download the installer. 
   - Install PuTTY by following the setup instructions.
  
2. **Open PuTTY**:  
   - Open PuTTY from your Start Menu.

3. **Configure PuTTY**:  
   - In the **Host Name (or IP address)** field, type `openscan.local`.
   - In the **Port** field, ensure `22` is set (it’s the default SSH port).
   - Click **Open**.

4. **Log in**:  
   - When prompted, enter your **username** (`pi`) and **password** (`raspberry` or your custom one).

Now you're connected to your Raspberry Pi!

---

## 2. Linux
Most Linux distributions come with SSH pre-installed, and accessing the Raspberry Pi is easy.

### Steps:
1. **Open the Terminal**:  
   - Use `Ctrl+Alt+T` to open the terminal.

2. **Enter the SSH command**:  
   - Type the following command:
     ```bash
     ssh pi@openscan.local
     ```
   - Press **Enter**.

3. **Enter the password**:  
   - When prompted, type your Raspberry Pi password (`raspberry` or your custom one).

You’re now connected!

---

## 3. macOS
macOS also comes with SSH built-in, so no extra software is needed.

### Steps:
1. **Open Terminal**:  
   - Press `Cmd + Space`, type "Terminal", and hit **Enter**.

2. **Enter the SSH command**:  
   - Type:
     ```bash
     ssh pi@openscan.local
     ```
   - Press **Enter**.

3. **Enter the password**:  
   - When prompted, type your Raspberry Pi password (`raspberry` or your custom one).

You’re now connected to the Raspberry Pi.

---

## Additional Notes:
- **First Time Connecting**:  
   When you connect to your Raspberry Pi for the first time, you might see a warning about the authenticity of the host. Just type `yes` and press Enter.

- **Password Security**:  
   If the password was changed from the default (`raspberry`), make sure to provide the correct one.

---
