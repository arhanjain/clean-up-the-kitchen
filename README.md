# clean-up-the-kitchen

1. Clone this repo!
2. Create conda/mamba environment from `env.yml` file: `conda env create -f env.yml` or `micromamba env create -f env.yml`
    - This step is untested, if this doesn't work contact @arhanjain
3. Make a `data` folder and add the *usdz* file sent by @arhanjain into the `data` directory
4. Run `python scripts/xform_mapper.py --usd_path [PATH_TO_USDZ]`
5. Open `config/config.yml` and replace the **usd_info_path** with the output path of file produced by step 4.
6. Run `python collect.py --task Real2Sim`

**If you reach this point, you have now at least begun data collection successfully. For next steps, reach out to Arhan.**

