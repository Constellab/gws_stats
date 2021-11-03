# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import click
import sys
import os
import shutil
import json
import re

__cdir__ = os.path.dirname(os.path.abspath(__file__))

@click.command(context_settings=dict(
    ignore_unknown_options=False,
    allow_extra_args=False
))
@click.pass_context
@click.option('--name', '-n', help='Brick name')
@click.option('--desc', '-d', help='Brick description')
def main(ctx, name=None, desc=None):
    name = name.lower().replace(" ", "_").replace("-", "_")

    skeleton_dir = os.path.join(__cdir__, "../../skeleton")
    dest_dir = os.path.join(__cdir__, "../../", name)

    if os.path.exists(dest_dir):
        raise Exception("A brick with the same name already exist")
    
    shutil.copytree(
        skeleton_dir, 
        dest_dir,
        dirs_exist_ok=True
    )
    
    if os.path.exists(os.path.join(dest_dir, name)):
        shutil.rmtree(dest_dir)
        raise Exception(f"The brick name '{name}' is not valid")
        
    shutil.move(
        os.path.join(dest_dir, "src", "skeleton"), 
        os.path.join(dest_dir, "src", name)
    )

    # remove .git folder
    shutil.rmtree(os.path.join(dest_dir, ".git"))
    update_settings_file(dest_dir, name)
    update_app(dest_dir, name)
    udpate_readme(dest_dir, name)
    udpate_setup(dest_dir, name, desc)
    print("The brick was successfully created.")


def update_settings_file(dest_dir, name):
    """ Update settings.json """
    settings_file = os.path.join(dest_dir, "settings.json")
    with open(settings_file, 'r') as f:
        settings = json.load(f)
        settings["name"] = name
        if "app_dir" in settings:
            del settings["app_dir"]
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=4)
    #replace all words 'skeleton' in settings.json
    with open(settings_file, 'r') as f:
        text = f.read()
        text = text.replace("skeleton", name)
    with open(settings_file, 'w') as f:
        f.write(text)

def update_app(dest_dir, name):
    """ Replace all words 'skeleton' in app.py """
    file = os.path.join(dest_dir, "src", name, "./app.py")
    with open(file, 'r') as f:
        text = f.read()
        text = text.replace("skeleton", name)
    with open(file, 'w') as f:
        f.write(text)

def udpate_readme(dest_dir, name):
    """ Replace all words 'skeleton' in README.md """
    file = os.path.join(dest_dir, "./README.md")
    with open(file, 'r') as f:
        text = f.read()
        text = text.replace("skeleton", name)
        text = text.replace("Skeleton", name.title())
    with open(file, 'w') as f:
        f.write(text)

def udpate_setup(dest_dir, name, desc):
    """ Update setup.py file """
    file = os.path.join(dest_dir, "./setup.py")
    with open(file, 'r') as f:
        text = f.read()
        text = text.replace("skeleton", name)
        text = re.sub(
                r"DESCRIPTION\s*\=.*", 
                f"DESCRIPTION = \"{desc}\"", 
                text
            )
    with open(file, 'w') as f:
        f.write(text)

if __name__ == "__main__":
    main()