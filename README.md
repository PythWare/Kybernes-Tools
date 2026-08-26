# Kybernes Tools

This repository will hold GUI Modding Tools I build to be used for PC Koei Tecmo/Omega Force games. They're meant to be used with GokonSoftworks unless listed as standalone, this repository will be updated periodically. As of March 29 2026 only Steel Editor, Bubble Editor, Harklight, Wild Liberd, Kybernes Scanner, Festum Conversion are added but future Editors/Tools will be added here at later dates.

# Requirements to use my tools

Only a Python 3 installation, if a tool needs something like Pillow (python imaging library) then the tool's section will say it in the readme.

# Harklight, KVS Audio Tool

Harklight is meant to be used with GokonSoftworks (since it handles subcontainer rebuilding which a lot of KVS files are stored in) but it has standalone usage. Harklight can decrypt KVS files to playable Oggs, convert Oggs to valid KVS files, and has a rad custom GUI all done in Python. It supports single or batch usage.

<img width="1909" height="1032" alt="kv1" src="https://github.com/user-attachments/assets/f50e7db1-c3a4-4959-af3e-579cb1d7303b" />
<img width="1909" height="1037" alt="kv2" src="https://github.com/user-attachments/assets/d3d5314e-9cec-4dbb-a70c-dad02d844958" />
<img width="1914" height="1035" alt="kv3" src="https://github.com/user-attachments/assets/ebb33ffe-1f26-408c-b8c9-a9984fcfa2db" />
<img width="1907" height="1036" alt="kv4" src="https://github.com/user-attachments/assets/41cc1d39-98e3-4bff-9a1d-d4e5224a3b99" />

# Uses for Harklight

KVS is for a lot of Koei Tecmo games the encrypted ogg format they use for voiced audio and other various things like bgm, sounds, etc. This tool is free and written in Python with no dependencies beyond having a python 3 installation.

# Audio modding with Harklight

Use Harklight for converting KVS to ogg and vice versa, then when you need to apply your new audio mods use Aldnoah Engine to rebuild the KVS subcontainers. If you're dealing with loose KVS files that were not part of a subcontainer then you don't have to use AE's subcontainer rebuilding

# Wild Liberd, G1L Tool

Wild Liberd is a Standalone GUI batch (can scan subdirectories too) G1L Unpacker/Repacker for G1L files that store KOVS/KTSS files, tested on Warriors Orochi 3 (I replaced BGMS with custom ones from my favorite singers/anime opening songs), Dynasty Warriors 7 XL, Toukiden Kiwami. I don't guarentee it works for every Koei Tecmo game, sometimes Omega Force stores other audio formats within G1L containers but I know it works for Warriors Orochi 3, Toukiden Kiwami, and Dynasty Warriors 7 XL. If you try it on other games, it should unpack without issue unless it detects a signature that isn't KOVS/KTSS since Wild Liberd is in an early state. I'll continue updating it to support other formats (for example, Bladestorm Nightmare has files with RIFF signatures in some of the G1L files so i'll need to add support for that later on).

Wild Liberd supports dynamic file sizes for user songs, you don't have to have the same file size as the bgms you want to replace. Your music can be smaller/larger than the original KOVS/KTSS files.

<img width="911" height="645" alt="k1" src="https://github.com/user-attachments/assets/796af5b4-86cd-4936-882b-4d36b7e640b3" />

<img width="910" height="647" alt="k2" src="https://github.com/user-attachments/assets/897e48e2-a6f9-4b02-8776-cca7a81f309c" />

<img width="912" height="647" alt="u40" src="https://github.com/user-attachments/assets/e02b976b-db62-4691-99f4-afd3c6844ef6" />

<img width="915" height="654" alt="u41" src="https://github.com/user-attachments/assets/3c7a55aa-f859-4da9-adcd-993acad31ec2" />

<img width="918" height="650" alt="wl1" src="https://github.com/user-attachments/assets/ac04e054-2dba-49b9-83e0-7e245cc4d03d" />

<img width="920" height="647" alt="wl2" src="https://github.com/user-attachments/assets/e1001230-98f6-4add-ae88-4b8078ce8804" />

# Uses For Wild Liberd

Wild Liberd is good if you want to replace BGMS with your own music, you could replace every file within the G1L with your chosen music and the game will load it. Review Audio Modding section.

# Audio Modding With Wild Liberd

Wild Liberd Unpacks/Repacks G1L files but you still have to convert your music you want to a format the game expects which in orochi 3's case and any other G1L format that stores KOVS/KTSS files, is KOVS/KTSS. Use Harklight for KOVS/KVS usage.

To use with Wild Liberd, convert the songs you want to KVS with Harklight and place them in the unpacked folder of the G1L that you want to repack but your songs must be named after the original KVS files you want to replace. You need to replace the KVS files with yours with matching names (i.e., if I want animesong.ogg to be played ingame then I need to convert to KVS with kvs2ogg and then replace 00000.kvs in the G1L folder with animesong.kvs renamed to 00000.kvs). Before clicking repack, select the G1L file you want to repack (listed in the GUI as "Original G1L File". The number of .kvs files must match original toc_count (meaning if the G1L unpacks with 226 files, you must only repack with the same amount of files). .kvs files must be named 00000.kvs, 00001.kvs, etc (5 digit names).

# Kybernes Scanner

Kybernes Scanner is a GUI WBD/WBH tool meant to be used with GokonSoftworks for wrapped Koei Tecmo Wave Bank WBD/WBH files as of version 0.6 of Kybernes Tools, meaning it's meant to be used with files that store the WBD/WBH as a single combined file (like Warriors Orochi 3's case). It unpacks the wrapped files, unpacks the subsongs/subaudio from the WBD files, and creates wav versions for you to preview. It also allows rebuilding the files with the correct codec (PCM/MSADPCM/DSP), offsets, and metadata so the game loads it. Support for dynamic file size (meaning your replacement wav files can be larger or smaller than the originals) is implemented.

<img width="916" height="659" alt="k3" src="https://github.com/user-attachments/assets/0a9a76a4-8fe9-4be1-9576-8438b6066507" />

<img width="915" height="658" alt="k4" src="https://github.com/user-attachments/assets/a091633b-637d-4672-be76-87daec2db4de" />

# Kybernes Scanner Guide

Replace audio files for WBD/WBH wrapped files:

Replace any ####.wav you want (keep the same filename).

Your replacement must be:

WAV → PCM_S16LE (16-bit PCM/signed 16-bit/uncompressed)

That’s the requirement. Good labels you might see in converters/editors:

PCM_S16LE, 16-bit PCM, Signed 16-bit PCM, and WAV (Microsoft PCM) 16-bit.

Avoid these (don’t use them):

Microsoft ADPCM, IMA ADPCM (compressed WAV), MP3, AAC, OGG, FLAC, and 32-bit float WAV.

Don’t worry about sample rate/channels, you can use any sample rate (44.1k/48k/etc) and mono/stereo. The tool automatically converts to whatever the original sound expects (it reads the original settings from the WBH and re-encodes correctly).

Repack Guide:

Click Repack. The tool rebuilds the WBD and WBH and outputs a new wrapped bank .bin. Use Aldnoah Engine's Mod Manager to apply/disable mods.

# Steel/Bubble Unit Editors for Warriors Orochi 3

These standalone editors mod warriors orochi 3's LINKFILE_000.BIN file, both editors don't require the game to be unpacked. Make sure to read the text files within the Editors folder for how to use. As time goes on more fields will be discovered and i'll update the editors with more features/field names.

Credit goes to Michael for identifying some of the fields in the Unit Data, their lengths, and finding the Unit Names. Credit also goes to KanbeiKS7 for giving me permission to include documentation files of values (WO3DE_Models, WO3DE_Moveset, WO3DE_Voices).
Credit also goes to LunaMeiya for identifying the Costume ID field.

Steel Editor has more features than Bubble Editor for now until I have time to work on the Bubble Editor, the feature that Steel Editor has over Bubble Editor is fields can have names displayed with the values. I'll add this to Bubble Editor at a later date.

<img width="1919" height="1034" alt="bu1" src="https://github.com/user-attachments/assets/c2c7629f-2cfc-4efa-9f0b-ab8c53f7e1a0" />

<img width="1916" height="1041" alt="bu2" src="https://github.com/user-attachments/assets/e8dab084-ff1e-4bcd-832e-6c47482a542a" />

<img width="1907" height="1033" alt="bu3" src="https://github.com/user-attachments/assets/2c61cf65-45e2-4f7d-af60-aa457fd9d681" />

<img width="1907" height="1036" alt="bu4" src="https://github.com/user-attachments/assets/b0511fb1-36fd-4a6d-b751-5c638b0c2259" />

<img width="1909" height="1035" alt="st1" src="https://github.com/user-attachments/assets/1d18df7b-f653-46ea-8c11-9c5b2b3abf74" />

<img width="1910" height="1032" alt="st2" src="https://github.com/user-attachments/assets/2cec2112-29dc-4aa5-b1b5-cc13543e649f" />

<img width="1913" height="1031" alt="st3" src="https://github.com/user-attachments/assets/caaa276c-aa70-40ec-ab71-4320d692f9d6" />

<img width="1918" height="1044" alt="st4" src="https://github.com/user-attachments/assets/10fac47f-b7e4-47b9-9011-dae4d697d567" />

<img width="1912" height="1034" alt="st5" src="https://github.com/user-attachments/assets/52f1fd7b-01ce-425b-9c2f-f634ebcb4ca0" />

# Festum Conversion

A GUI Binary Translating Tool that can be used for XL, ECB, EM, MESC, etc formats that Koei Tecmo uses for storing strings/text. You can either translate within the tool or export/import json files. It can also be used for string modding if you don't want to translate but instead want to change the names of things (i.e., changing Lu Bu to Rad Bu as an example). One of the benefits of using Festum Conversion is it doesn't restrict you to the original string length limit the games expect, you can translate without string length limits in the binary file. So essentially, if your translation is longer than the original text then you can still use the longer translation without issue. You are not restricted to the original byte length limits the games impose.

To clarify though, while you can translate or mod strings to be however long you desire, you still have to keep in mind the games' font and text scaling as well as character encodings (meaning if you wanted to translate the English version to say Arabic or Vietnamese as an example, you need to make sure the game you're translating supports the character encoding your language requires such as UTF-8 and other various ones). Festum Conversion provides you an easy way to translate binary files or mod strings without being held back by byte length limits.

<img width="1181" height="775" alt="fe1" src="https://github.com/user-attachments/assets/398b6741-8fce-446a-9b21-858a989f77f9" />

<img width="1177" height="772" alt="fe3" src="https://github.com/user-attachments/assets/cb642c20-9891-44c7-8b4a-44d0b39145cd" />

<img width="1279" height="754" alt="fe8" src="https://github.com/user-attachments/assets/07c361f3-37b6-45ad-9a5b-bbd01977b2ab" />

<img width="1277" height="751" alt="fe9" src="https://github.com/user-attachments/assets/38af068e-836c-476c-8a81-11cc7ad7c84e" />

# References

The names of the tools are references to my favorite mecha animes Aldnoah Zero, Argevollen, and Fafner. Rad animes!

# Future Tools

G1T Krieger

<img width="1113" height="788" alt="k11" src="https://github.com/user-attachments/assets/20070dd3-13b0-4ed5-990a-c107a5a1a507" />

<img width="1108" height="783" alt="k16" src="https://github.com/user-attachments/assets/dbd76210-1830-4e6f-9579-4b8695f3a345" />
