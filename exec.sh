#!/usr/bin/env bash

cd ~/CoppeliaSim_Edu_V4_6_0_rev16_Ubuntu22_04/ \
&& ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23002 -GwsRemoteApi.port=23051 ~/code_workspace/Husky_CS_SB3/HuskyModels/DataGen.ttt
#&& ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23002 -GwsRemoteApi.port=23051 ~/code_workspace/Husky_CS_SB3/HuskyModels/HuskyInfinityGym.ttt \
