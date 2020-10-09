docker buildx create --name mybuilder \
&& docker buildx use mybuilder \
&& docker buildx inspect --bootstrap \
&& docker buildx build --platform linux/arm/v7 -t rio05docker/covid_continuos_learning/mlflowui_arm . --push
