#version 330 core
layout(location = 0) in vec2 position;
out vec4 fragPosition;  // Variable de sortie qui va stocker la position

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
	fragPosition = gl_Position;
}
