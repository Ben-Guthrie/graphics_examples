from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
from framework import *

vertex_dim = 4
num_points = None
unitvect = np.array([1., 2., 0.5], dtype="float32")
unitvect = unitvect / np.linalg.norm(unitvect)
theta = 0.0
vQuat = None

fovy = 45.
aspect = None
zNear = 0.5
zFar = 3.0

zCamera = 1.5
zAdd = 0.01

view_reference_point = np.array([0., 0., zCamera])
view_plane_normal = np.array([0., 0., 1.])
view_up_vector = np.array([0., 1., 0.])

vModelView = None
vPerspective = None

# String containing fragment shader program written in GLSL
strFragmentShader = """
#version 330

out vec4 outputColor;
void main()
{
    outputColor = vec4(0.5, 0.5, 0.5, 0.5);
}
"""

def get_quaternion(unit_vector, phi):
    """Get the quaternion from the e unit vector and rotation angle."""
    assert unit_vector.shape == (3, )
    assert isinstance(phi, (float, np.ndarray))
    phi = np.atleast_1d(phi)
    q = np.zeros((len(phi), 4))
    q[:, :3] = np.sin(phi/2.)[:, np.newaxis] * unit_vector[np.newaxis, :]
    q[:, 3] = np.cos(phi/2.)
    return np.squeeze(q)


def get_cube_vertices(width=1.):
    assert isinstance(width, (int, float))
    vertices = np.array([[-.5, -.5, 0.5, 1.],
                         [-.5, 0.5, 0.5, 1.],
                         [0.5, 0.5, 0.5, 1.],
                         [0.5, -.5, 0.5, 1.],
                         [-.5, -.5, -.5, 1.],
                         [-.5, 0.5, -.5, 1.],
                         [0.5, 0.5, -.5, 1.],
                         [0.5, -.5, -.5, 1.]],
                        dtype="float32")
    vertices[:, :3] = vertices[:, :3] * width

    faces = np.array([[1, 0, 3, 2],
                      [2, 3, 7, 6],
                      [3, 0, 4, 7],
                      [6, 5, 1, 2],
                      [4, 5, 6, 7],
                      [5, 4, 0, 1]])

    num_faces, num_indices = faces.shape
    # Number of indices required for triangles
    num_indices = (num_indices - 2) * 3

    points = np.zeros(((num_faces * num_indices), 4), dtype="float32")
    for i in range(num_faces):
        points[i*num_indices:(i+1)*num_indices] = quad(faces[i], vertices)

    return points

def quad(face, vertices):
    """Construct a face by partitioning 4 vertices into 2 triangles."""
    assert face.shape == (4, )
    a, b, c, d = face
    indices = [a, b, c, a, c, d]
    points = vertices[indices]
    return points


def get_perspective(fov_y, aspect_ratio, near, far):
    mat = np.zeros((4, 4), dtype="float32")
    top = near * np.tan(fov_y)
    right = top * aspect_ratio
    mat[0, 0] = float(near) / right
    mat[1, 1] = float(near) / top
    mat[2, 2] = -float(far + near) / (far - near)
    mat[2, 3] = -1
    mat[3, 2] = -2.*far*near / (far - near)

    return mat


def get_model_view(vrp, vpn, vup):
    v = vup - np.dot(vup, vpn) / np.dot(vpn, vpn) * vpn
    v = v / np.linalg.norm(v)
    n = vpn / np.linalg.norm(vpn)

    return get_model_view_from_unitvectors(vrp, n, v)


def get_model_view_from_unitvectors(p, n, v):
    """n and v must be unit vectors and must be orthogonal"""
    u = np.cross(v, n)

    mat = np.zeros((4, 4), dtype="float32")
    mat[:3,:3] = np.vstack((u, v, n))
    mat[:3, 3] = np.dot(-np.vstack((u, v, n)), p)
    mat[3, 3] = 1.
    return mat


def init():
    width = 1.
    points = get_cube_vertices(width)
    global num_points
    num_points = points.shape[0]
    points = points.flatten()

    # Create the shaders and program
    shader_list = []
    shader_list.append(loadShader(GL_VERTEX_SHADER, "rotating_quat.vert"))
    shader_list.append(createShader(GL_FRAGMENT_SHADER, strFragmentShader))
    program = createProgram(shader_list)
    for shader in shader_list:
        glDeleteShader(shader)
    glUseProgram(program)

    # Link quaternion to the shader
    global vQuat
    vQuat = glGetUniformLocation(program, "vQuat")
    quaternion = get_quaternion(unitvect, theta)
    glUniform4fv(vQuat, 1, quaternion)

    # Link model view and perspective matrices
    global vPerspective, vModelView
    vPerspective = glGetUniformLocation(program, "vPerspective")
    vModelView = glGetUniformLocation(program, "vModelView")


    # Setup the vertex buffer for storing vertex coordinates
    position_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, position_buffer)
    glBufferData(GL_ARRAY_BUFFER, points, GL_STATIC_DRAW)

    # Create a vertex array object
    glBindVertexArray(glGenVertexArrays(1))

    # Initialize the vertex position attribute from the vertex shader
    loc = glGetAttribLocation(program, "vPosition")
    glEnableVertexAttribArray(loc)
    glVertexAttribPointer(loc, vertex_dim, GL_FLOAT, GL_FALSE, 0, None)

    # Enable depth test and cull invisible faces
    """
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glFrontFace(GL_CW)
    """

    glEnable(GL_DEPTH_TEST)
    glDepthMask(GL_TRUE)
    glDepthFunc(GL_LEQUAL)
    glDepthRange(0.0, 0.5 * width)
    glClearColor(0., 0., 0., 0.)


def display():
    """Draw the points on the display."""
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Calculate and send quaternion
    global theta, quaternion
    theta += 0.01
    if theta > 2*np.pi:
        theta = 0.0
    quaternion = get_quaternion(unitvect, theta)
    glUniform4fv(vQuat, 1, quaternion)

    # Get model view and perspective matrices
    global zCamera, zAdd, view_reference_point
    zCamera += zAdd
    if zCamera > 4.0:
        zAdd = -0.01
    elif zCamera < 1.0:
        zAdd = 0.01
    view_reference_point[2] = zCamera
    model_view = get_model_view_from_unitvectors(view_reference_point, view_plane_normal,
                                                 view_up_vector)
    glUniformMatrix4fv(vModelView, 1, GL_TRUE, model_view)
    perspective = get_perspective(fovy, aspect, zNear, zFar)
    glUniformMatrix4fv(vPerspective, 1, GL_TRUE, perspective)

    # Draw the cube
    glDrawArrays(GL_TRIANGLES, 0, num_points)
    glutSwapBuffers()       # Because we are using double buffering
    glutPostRedisplay()


def reshape(w, h):
    """Reshape the window."""
    glViewport(0, 0, w, h)

    global aspect
    aspect = float(w) / h


def keyboard(key, x, y):
    """Exit when ESC is pressed."""
    if ord(key) == 27:
        glutLeaveMainLoop()
        return


def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE)
    glutInitWindowSize(640, 480)
    glutCreateWindow("simple OpenGL example")

    init()
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)

    glutMainLoop()


if __name__ == "__main__":
    main()
