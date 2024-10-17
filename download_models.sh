#!/bin/bash

# Create the models directory in Google Drive
mkdir -p models

echo "Downloading models from Dropbox into the 'models' directory..."

# Download models and check if the download was successful
if wget -O models/new_data.pt https://www.dropbox.com/scl/fi/9zf9x3w7r4rizmnn9cbk3/new_data.pt?rlkey=h5gnex1tc0i5egsjpoe3hct5l&st=2araenae&dl=1; then
    echo "new_data.pt downloaded successfully."
else
    echo "Failed to download new_data.pt"
fi

if wget -O models/Substitution.pt https://www.dropbox.com/scl/fi/638s1flkxaeey0vv2vlrg/Substitution.pt?rlkey=j4720865x3afz53yfzx9xmazo&st=kg4o6z75&dl=1; then
    echo "Substitution.pt downloaded successfully."
else
    echo "Failed to download Substitution.pt"
fi

if wget -O models/playershirt.pt https://www.dropbox.com/scl/fi/fmkdhhn8aas1jjr3l2xc3/playershirt.pt?rlkey=8kra62hs2fc36p677sm4ms32c&st=70rxlt6a&dl=1; then
    echo "playershirt.pt downloaded successfully."
else
    echo "Failed to download playershirt.pt"
fi

if wget -O models/old_data.pt https://www.dropbox.com/scl/fi/5wh4yy2ego497sw7ut01y/old_data.pt?rlkey=pkktrpl7kudux5xbaxu2is550&st=ftxxrz0d&dl=1; then
    echo "old_data.pt downloaded successfully."
else
    echo "Failed to download old_data.pt"
fi

echo "Download process completed."
