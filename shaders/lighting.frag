#version 330

uniform vec4 ambientProduct;
uniform vec4 diffuseProduct;
uniform vec4 specularProduct;
uniform float shininess;
in vec3 N, L, E;
out vec4 outputColor;

void main()
{
  vec4 fColor;

  vec3 H = normalize(L+E);
  vec4 ambient = ambientProduct;
  vec4 diffuse = max(dot(L, N), 0.0) * diffuseProduct;
  vec4 specular = max(pow(max(dot(N, H), 0.0), shininess) * specularProduct, 0.0);

  fColor = ambient + diffuse + specular;
  fColor.a = 1.0;

  outputColor = fColor;
}
