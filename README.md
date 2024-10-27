# clean-up-the-kitchen

**Teleoperation**

`python teleop.py --task Real2Sim --ds_name [DS_NAME] usd_path=[USDPATH] actions.type=[relative/absolute] --enable_cameras --headless`

**Collect Motion Planned Trajetories**

`todo`

**Rollout models in simulation**

`python collect.py --task Real2Sim usd_path=./data/basement_flat.usd data_collection.save=False actions.type="relative"`

**Rollout models in real-world**

`python deploy_real.py`




1. Clone this repo!
2. Create conda/mamba environment with `conda create -n real2sim python=3.10` or `micromamba create -n real2sim python=3.10`
3. Activate the environment: `conda activate real2sim`
4. Install dependencies using requirements file. `pip install -r ./requirements.txt`
5. Go to `https://github.com/arhanjain/M2T2`, clone it anywhere, and follow the README instructions there to install M2T2
6. Specify the path to `m2t2.pth` model weights in `config/config.yml` under **grasp.eval** section
7. Go to `https://curobo.org/get_started/1_install_instructions.html#install-for-use-in-isaac-sim` and install CuRobo
8. Make a `data` folder and add the *usdz* file sent by @arhanjain into the `data` directory
9. Run `python scripts/xform_mapper.py --usd_path [PATH_TO_USDZ]`
10. Open `config/config.yml` and replace the **usd_info_path** with the output path of file produced by step 4.
11. Run `python collect.py --task Real2Sim`
12. Convert your data to hdf5 format using `python scripts/convert_to_hdf5.py --data_dir [DATADIR]`

**If you reach this point, you have now at least begun data collection successfully. For next steps, reach out to Arhan.**


# Teleoperation

## NUC

### Running the Franka Server 

1. **Open 3 Terminal Shells**: This setup requires three separate terminal windows.
2. **Activate Conda Environment**: In each shell, activate the `arhan-droid` environment with:
    ```bash
    conda activate arhan-droid
    ```
3. **Navigate to Project Directory**: Change directory to `~/droid/` in each shell:
    ```bash
    cd ~/droid/
    ```
4. **Run Commands in Order**:
   - In the first shell, run:
     ```bash
     ./droid/franka/launch_robot.py
     ```
   - In the second shell, run:
     ```bash
     ./droid/franka/launch_gripper.py
     ```
   - In the third shell, run:
     ```bash
     ./scripts/server/launch_server.py
     ```
    > **Hint**: Ensure that both the robot and the gripper are shown as connected in the launch robot/gripper scripts!

## Control PC

The **Control PC** section covers the setup steps specific to the PC used for controlling the teleoperation system.

1. **Activate Conda Environment**: Open a terminal and activate the `auto-articulate` environment:
    ```bash
    conda activate auto-articulate
    ```
2. **Connect Oculus**: Ensure the Oculus device is properly connected to the Control PC.
3. **Verify Connection**: Run the following command to check that the Oculus device is recognized:
    ```bash
    adb devices
    ```
    You should see the Oculus device listed as "attached" if it’s connected correctly.
4. **Navigate to Project Directory**: Change directory to `~/projects/clean-up-the-kitchen/`:
    ```bash
    cd ~/projects/clean-up-the-kitchen/
    ```
5. **Run Teleoperation Script**: Start the teleoperation process by running:
    ```bash
    python ./scripts/teleop_real [DATASET_NAME]
    ```
   Replace `[DATASET_NAME]` with your desired dataset name to save the session’s data.

---

