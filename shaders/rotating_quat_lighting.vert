#version 330

in vec4 vPosition;
uniform vec4 vQuat;
uniform mat4 vModelView;
uniform mat4 vPerspective;

in vec3 vNormal;
out vec3 N, L, E;
uniform vec4 lightDirection;

vec4 quat_conj(vec4 q)
{
  return vec4(-q.x, -q.y, -q.z, q.w);
}

vec4 quat_mult(vec4 q1, vec4 q2)
{
  vec4 qr;
  qr.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
  qr.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
  qr.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
  qr.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);
  return qr;
}

vec4 rotate_by_q(vec4 q, vec3 r)
{
  vec4 q_conj = quat_conj(q);
  vec4 q_pos = vec4(r.x, r.y, r.z, 0);
  vec4 q_tmp = quat_mult(vQuat, q_pos);
  return quat_mult(q_tmp, q_conj);
}

void main()
{
  // Rotation
  vec4 q_pos = rotate_by_q(vQuat, vPosition.xyz);
  q_pos.w = vPosition.w;

  // Rotate normals
  vec4 normal = rotate_by_q(vQuat, vNormal);

  // Lighting
  N = normalize(normal.xyz);
  // L = normalize(lightPosition - vPosition).xyz;
  L = normalize(lightDirection.xyz);
  E = -normalize(vPosition.xyz);

  // Position
  gl_Position = vPerspective * vModelView * q_pos;
}
