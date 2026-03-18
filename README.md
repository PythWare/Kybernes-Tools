# Kybernes Tools

This repository will hold GUI Modding Tools I build to be used for PC Koei Tecmo/Omega Force games. They're meant to be used with Aldnoah Engine unless listed as standalone, this repository will be updated periodically. As of March 18 2026 only Harklight, Wild Liberd, and Kybernes Scanner are added but U-Link System Stage Editor (Stage/Battlefield Editor for Dynasty Warriors 8 XL, Dynasty Warriors 7 XL, Dynasty Warriors 6, Samurai Warriors 2), Silver Will (Unit Editor for Bladestorm), U-Link System (unit editor for Orochi 3), Festum Converson (translation/string editing tool), G1T Krieger (G1T Viewer/Converter tool), and other Editors/Tools will be added here at later dates.

# Harklight, KVS Audio Tool

Harklight is meant to be used with Aldnoah Engine (since AE handles subcontainer rebuilding which a lot of KVS files are stored in) but it has standalone usage. Harklight can decrypt KVS files to playable Oggs, convert Oggs to valid KVS files, and has a rad custom GUI all done in Python. It supports single or batch usage.

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

Kybernes Scanner is a GUI WBD/WBH tool meant to be used with Aldnoah Engine for wrapped Koei Tecmo Wave Bank WBD/WBH files as of version 0.6 of Kybernes Tools, meaning it's meant to be used with files that store the WBD/WBH as a single combined file (like Warriors Orochi 3's case). It unpacks the wrapped files, unpacks the subsongs/subaudio from the WBD files, and creates wav versions for you to preview. It also allows rebuilding the files with the correct codec (PCM/MSADPCM/DSP), offsets, and metadata so the game loads it. Support for dynamic file size (meaning your replacement wav files can be larger or smaller than the originals) is implemented.

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


# References

The names of the tools are references to my favorite mecha animes Aldnoah Zero, Argevollen, and Fafner. Rad animes!

# Future Tools

U-Link System Stage Editor
<img width="1234" height="644" alt="u21" src="https://github.com/user-attachments/assets/4a7e803e-2abe-45ae-ad94-73e0c5550678" />
<img width="1089" height="725" alt="u23" src="https://github.com/user-attachments/assets/f7b38d12-bc9b-4962-91ec-260ccd6f2c1b" />
<img width="1710" height="729" alt="u33" src="https://github.com/user-attachments/assets/d1ed5e7a-4cf8-469b-b55e-c3e2ab3922e5" />
<img width="949" height="526" alt="u38" src="https://github.com/user-attachments/assets/6d31722e-1996-415a-82a2-ae48287b9f56" />
<img width="1095" height="724" alt="u39" src="https://github.com/user-attachments/assets/48674be1-5e4c-41fc-b60d-085cced81ed6" />



Festum Conversion

<img width="991" height="708" alt="f1" src="https://github.com/user-attachments/assets/6d7eab79-bda4-47d7-9c0d-5603b6779c11" />

<img width="984" height="702" alt="f2" src="https://github.com/user-attachments/assets/a9b17076-80b3-43bf-8f0e-24c7c039664e" />

G1T Krieger

<img width="1113" height="788" alt="k11" src="https://github.com/user-attachments/assets/20070dd3-13b0-4ed5-990a-c107a5a1a507" />

<img width="1108" height="783" alt="k16" src="https://github.com/user-attachments/assets/dbd76210-1830-4e6f-9579-4b8695f3a345" />


U-Link System

<img width="1008" height="826" alt="a7" src="https://github.com/user-attachments/assets/035cbe70-9528-44b8-b43a-c1b5b9ece12a" />

Silver Will

<img width="796" height="625" alt="a6" src="https://github.com/user-attachments/assets/320bb5a2-23b3-4452-864e-5b67dad6f63b" />
