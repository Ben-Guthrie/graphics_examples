from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np
from framework import *

vertex_dim = 4
num_points = None
theta = np.array([0., 0., 0.], dtype="float32")

# String containing vertex shader program written in GLSL
strVertexShader = """
#version 330

in vec4 vPosition;

void
main()
{
    gl_Position = vPosition;
}
"""

# String containing fragment shader program written in GLSL
strFragmentShader = """
#version 330

out vec4 outputColor;
void main()
{
    outputColor = vec4(1.0, 0.0, 0.0, 1.0);
}
"""

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


def init():
    width = 1.
    points = get_cube_vertices(width)
    global num_points
    num_points = points.shape[0]
    points = points.flatten()
    print num_points
    print points
    """
    global num_points
    num_points = 36
    # points = np.array([0.5, 0.5, -0.5, 1.0,
                       # 0.5, -0.5, -0.5, 1.0,
                       # -0.5, -0.5, 0.5, 1.0,
                       # -0.5, -0.5, -0.5, 1.0,
                       # -0.5, 0.5, 0.5, 1.0,
                       # 0.5, 0.5, 0.5, 1.0], dtype="float32")
    points = np.array([-0.5, 0.5, 0.5, 1., -0.5, -0.5, 0.5, 1.,  0.5, -0.5, 0.5, 1., -0.5, 0.5, 0.5,
                       1.,  0.5, -0.5, 0.5, 1., 0.5, 0.5, 0.5, 1., 0.5, 0.5, 0.5, 1., 0.5, -0.5,
                       0.5, 1., 0.5, -0.5, -0.5, 1., 0.5, 0.5, 0.5, 1., 0.5, -0.5, -0.5, 1., 0.5,
                       0.5, -0.5, 1., 0.5, -0.5, 0.5, 1., -0.5, -0.5, 0.5, 1., -0.5, -0.5, -0.5, 1.,
                       0.5, -0.5, 0.5, 1., -0.5, -0.5, -0.5, 1., 0.5, -0.5, -0.5, 1., 0.5, 0.5, -0.5,
                       1., -0.5, 0.5, -0.5, 1., -0.5, 0.5, 0.5, 1., 0.5, 0.5, -0.5, 1., -0.5, 0.5,
                       0.5, 1., 0.5, 0.5, 0.5, 1., -0.5, -0.5, -0.5, 1., -0.5, 0.5, -0.5, 1., 0.5,
                       0.5, -0.5, 1., -0.5, -0.5, -0.5, 1., 0.5, 0.5, -0.5, 1., 0.5, -0.5, -0.5, 1.,
                       -0.5, 0.5, -0.5, 1., -0.5, -0.5, -0.5, 1., -0.5, -0.5, 0.5, 1., -0.5, 0.5, -0.5,
                       1., -0.5, -0.5, 0.5, 1., -0.5, 0.5, 0.5, 1.], dtype="float32")
    width = 1.0
    print points
    """

    # Create the shaders and program
    shader_list = []
    shader_list.append(createShader(GL_VERTEX_SHADER, strVertexShader))
    shader_list.append(createShader(GL_FRAGMENT_SHADER, strFragmentShader))
    program = createProgram(shader_list)
    for shader in shader_list:
        glDeleteShader(shader)
    glUseProgram(program)

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
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glFrontFace(GL_CW)

    glEnable(GL_DEPTH_TEST)
    glDepthMask(GL_TRUE)
    glDepthFunc(GL_LEQUAL)
    glDepthRange(0.0, 0.5 * width)
    glClearColor(1., 1., 1., 1.)


def display():
    """Draw the points on the display."""
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glDrawArrays(GL_TRIANGLES, 0, num_points)
    assert glGetError() == GL_NO_ERROR
    glFlush()       # Because we are using double buffering


def reshape(w, h):
    """Reshape the window."""
    glViewport(0, 0, w, h)


def keyboard(key, x, y):
    """Exit when ESC is pressed."""
    if ord(key) == 27:
        glutLeaveMainLoop()
        return


def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH)
    glutInitWindowSize(640, 480)
    glutCreateWindow("simple OpenGL example")

    init()
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)

    glutMainLoop()


if __name__ == "__main__":
    main()
