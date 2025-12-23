# Kybernes Tools

This repository will hold GUI Modding Tools I build to be used for Koei Tecmo/Omega Force games. They're meant to be used with Aldnoah Engine unless listed as standalone, this repository will be updated periodically. As of December 22 2025 only Wild Liberd is added but Kybernes Scanner (used for WBD/WBH wrapped files such as in Orochi 3's case or loose paired WBD/WBH files like for Bladestorm Nightmare), Silver Will (Unit Editor for Bladestorm), U-Link System (unit editor for Orochi 3), Festum Converson (translation/string editing tool), and other Editors/Tools will be added here at later dates.

# Wild Liberd, G1L Tool

Wild Liberd is a Standalone GUI batch (can scan subdirectories too) G1L Unpacker/Repacker for G1L files that store KOVS/KTSS files, tested on Warriors Orochi 3 (I replaced BGMS with custom ones from my favorite singers/anime opening songs). I don't guarentee it works for every Koei Tecmo game, sometimes Omega Force stores other audio formats within G1L containers but I know it works for Warriors Orochi 3. If you try it on other games, it should unpack without issue unless it detects a signature that isn't KOVS/KTSS since Wild Liberd is in an early state. I'll continue updating it to support other formats (for example, Bladestorm Nightmare has files with RIFF signatures in some of the G1L files so i'll need to add support for that later on).

Wild Liberd supports dynamic file sizes for user songs, you don't have to have the same file size as the bgms you want to replace. Your music can be smaller/larger than the original KOVS/KTSS files.

<img width="911" height="645" alt="k1" src="https://github.com/user-attachments/assets/796af5b4-86cd-4936-882b-4d36b7e640b3" />

<img width="910" height="647" alt="k2" src="https://github.com/user-attachments/assets/897e48e2-a6f9-4b02-8776-cca7a81f309c" />

# Uses For Wild Liberd

Wild Liberd is good if you want to replace BGMS with your own music, you could replace every file within the G1L with your chosen music and the game will load it. Review Audio Modding section.

# Audio Modding With Wild Liberd

Wild Liberd Unpacks/Repacks G1L files but you still have to convert your music you want to a format the game expects which in orochi 3's case and any other G1L format that stores KOVS/KTSS files, is KOVS/KTSS. To do that, I recommend downloading kvs2ogg from the musou warriors discord server in the resources-and-other channel. kvs2ogg can convert mp3, wav, and ogg to kvs and vice versa.

To use with Wild Liberd, convert the songs you want to KVS with kvs2ogg and place them in the unpacked folder of the G1L that you want to repack but your songs must be named after the original KVS files you want to replace. You need to replace the KVS files with yours with matching names (i.e., if I want animesong.ogg to be played ingame then I need to convert to KVS with kvs2ogg and then replace 00000.kvs in the G1L folder with animesong.kvs renamed to 00000.kvs). Before clicking repack, select the G1L file you want to repack (listed in the GUI as "Original G1L File". The number of .kvs files must match original toc_count (meaning if the G1L unpacks with 226 files, you must only repack with the same amount of files). .kvs files must be named 00000.kvs, 00001.kvs, etc (5 digit names).

# References

The names of the tools are references to my favorite mecha animes Aldnoah Zero, Argevollen, and Fafner. Rad animes!

# Future Tools

Festum Conversion

<img width="991" height="708" alt="f1" src="https://github.com/user-attachments/assets/6d7eab79-bda4-47d7-9c0d-5603b6779c11" />

<img width="984" height="702" alt="f2" src="https://github.com/user-attachments/assets/a9b17076-80b3-43bf-8f0e-24c7c039664e" />

Kybernes Scanner

<img width="916" height="659" alt="k3" src="https://github.com/user-attachments/assets/0a9a76a4-8fe9-4be1-9576-8438b6066507" />

<img width="915" height="658" alt="k4" src="https://github.com/user-attachments/assets/a091633b-637d-4672-be76-87daec2db4de" />

U-Link System

<img width="1008" height="826" alt="a7" src="https://github.com/user-attachments/assets/035cbe70-9528-44b8-b43a-c1b5b9ece12a" />

Silver Will

<img width="796" height="625" alt="a6" src="https://github.com/user-attachments/assets/320bb5a2-23b3-4452-864e-5b67dad6f63b" />
