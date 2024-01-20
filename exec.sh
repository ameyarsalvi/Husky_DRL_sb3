#!/usr/bin/env bash

cd ~/CoppeliaSim_Edu_V4_6_0_rev16_Ubuntu22_04/ \
&& ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=23008 -GwsRemoteApi.port=23056 ~/code_workspace/Husky_CS_SB3/HuskyModels/HuskyInfinityGym.ttt
