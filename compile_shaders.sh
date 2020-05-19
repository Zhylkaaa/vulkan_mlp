#!/bin/bash

SHADER_DIR=shaders

shader_list=$(ls $SHADER_DIR | grep -E '.*\.comp$')

for shader_name in $shader_list; do
  glslc -o ${SHADER_DIR}/${shader_name}.spv ${SHADER_DIR}/${shader_name}
done