#!/usr/bin/env bash

cd ~/CoppeliaSim_Edu_V4_6_0_rev16_Ubuntu22_04/ \
&& ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=23004 -GwsRemoteApi.port=23053 ~/code_workspace/Husky_CS_SB3/HuskyModels/coneSharpPath.ttt
#&& ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23004 -GwsRemoteApi.port=23051 ~/code_workspace/Husky_CS_SB3/HuskyModels/cone_gymTrain.ttt
#&& ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23002 -GwsRemoteApi.port=23051 ~/code_workspace/Husky_CS_SB3/HuskyModels/HuskyInfinityGym.ttt \
