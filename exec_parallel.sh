#!/usr/bin/env bash

cd ~/CoppeliaSim_Edu_V4_6_0_rev16_Ubuntu22_04/

konsole --noclose --new-tab -e ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=23004 -GwsRemoteApi.port=23053 ~/code_workspace/Husky_CS_SB3/HuskyModels/HuskyInfinityGym.ttt && /bin/bash &
konsole --noclose --new-tab -e ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=23006 -GwsRemoteApi.port=23055 ~/code_workspace/Husky_CS_SB3/HuskyModels/HuskyInfinityGym.ttt && /bin/bash &
konsole --noclose --new-tab -e ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=23008 -GwsRemoteApi.port=23057 ~/code_workspace/Husky_CS_SB3/HuskyModels/HuskyInfinityGym.ttt && /bin/bash &
konsole --noclose --new-tab -e ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=23010 -GwsRemoteApi.port=23059 ~/code_workspace/Husky_CS_SB3/HuskyModels/HuskyInfinityGym.ttt && /bin/bash &
konsole --noclose --new-tab -e ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23012 -GwsRemoteApi.port=23061 ~/code_workspace/Husky_CS_SB3/HuskyModels/HuskyInfinityGym.ttt && /bin/bash &
#konsole --noclose --new-tab -e ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=23014 -GwsRemoteApi.port=23063 ~/code_workspace/Husky_CS_SB3/HuskyModels/GymHuskyVS.ttt && /bin/bash &
#konsole --noclose --new-tab -e ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=23016 -GwsRemoteApi.port=23065 ~/code_workspace/Husky_CS_SB3/HuskyModels/GymHuskyVS.ttt && /bin/bash &
#konsole --noclose --new-tab -e ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=23018 -GwsRemoteApi.port=23067 ~/code_workspace/Husky_CS_SB3/HuskyModels/GymHuskyVS.ttt && /bin/bash &
#konsole --noclose --new-tab -e ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=23020 -GwsRemoteApi.port=23069 ~/code_workspace/Husky_CS_SB3/HuskyModels/GymHuskyVS.ttt && /bin/bash &
#konsole --noclose --new-tab -e ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23022 -GwsRemoteApi.port=23071 ~/code_workspace/Husky_CS_SB3/HuskyModels/GymHuskyVS.ttt && /bin/bash
#konsole --noclose --new-tab -e ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23024 -GwsRemoteApi.port=23073 ~/code_workspace/Husky_CS_SB3/HuskyModels/GymHuskyVS.ttt && /bin/bash &
#konsole --noclose --new-tab -e ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23026 -GwsRemoteApi.port=23075 ~/code_workspace/Husky_CS_SB3/HuskyModels/GymHuskyVS.ttt && /bin/bash &
#konsole --noclose --new-tab -e ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23028 -GwsRemoteApi.port=23077 ~/code_workspace/Husky_CS_SB3/HuskyModels/GymHuskyVS.ttt && /bin/bash &
#konsole --noclose --new-tab -e ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23030 -GwsRemoteApi.port=23079 ~/code_workspace/Husky_CS_SB3/HuskyModels/GymHuskyVS.ttt && /bin/bash &
#konsole --noclose --new-tab -e ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23032 -GwsRemoteApi.port=23081 ~/code_workspace/Husky_CS_SB3/HuskyModels/GymHuskyVS.ttt && /bin/bash &
#konsole --noclose --new-tab -e ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23034 -GwsRemoteApi.port=23083 ~/code_workspace/Husky_CS_SB3/HuskyModels/GymHuskyVS.ttt && /bin/bash
