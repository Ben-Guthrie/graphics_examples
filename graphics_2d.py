import random
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *


num_points = 5000
vertex_dim = 2

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

def createShader(shader_type, shader_file):
    """Create and compile a shader."""
    shader = glCreateShader(shader_type)
    glShaderSource(shader, shader_file)

    glCompileShader(shader)

    status = None
    glGetShaderiv(shader, GL_COMPILE_STATUS, status)
    if status == GL_FALSE:
        strInfoLog = glGetShaderInfoLog(shader)
        strShaderType = ""
        if shader_type is GL_VERTEX_SHADER:
            strShaderType = "vertex"
        elif shader_type is GL_GEOMETRY_SHADER:
            strShaderType = "geometry"
        elif shader_type is GL_FRAGMENT_SHADER:
            strShaderType = "fragment"

        print "Compilation failure for " + strShaderType + " shader:\n" + strInfoLog

    return shader


def createProgram(shader_list):
    """Compiles shaders and returns the compiled program."""
    program = glCreateProgram()

    for shader in shader_list:
        glAttachShader(program, shader)

    glLinkProgram(program)

    status = glGetProgramiv(program, GL_LINK_STATUS)
    if status == GL_FALSE:
        str_info_log = glGetProgramInfoLog(program)
        print "Linker failure: \n" + str_info_log

    for shader in shader_list:
        glDetachShader(program, shader)

    return program

# Sierpinski gasket
def init():
    if vertex_dim == 2:
        vertices = np.array([[-1., -1.],
                             [0., 1.],
                             [1., -1.]])
    elif vertex_dim == 3:
        vertices = np.array([[-1., -1., 0.],
                             [0., 1., 0.],
                             [1., -1., 0.]])

    u = 0.5 * (vertices[0] + vertices[1])
    v = 0.5 * (vertices[0] + vertices[2])
    p = 0.5 * (u + v)

    points = [p]

    for i in range(1, num_points):
        j = random.randint(0, 2)
        p = 0.5 * (points[i-1] + vertices[j])
        points.append(p)
    points = np.array(points, dtype='float32').flatten()     # Must be a 1D array
    print points

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
    print loc
    glEnableVertexAttribArray(loc)
    glVertexAttribPointer(loc, vertex_dim, GL_FLOAT, GL_FALSE, 0, None)


def display():
    """Draw the points on the display."""
    glClearColor(0., 0., 0., 0.)
    glClear(GL_COLOR_BUFFER_BIT)
    glDrawArrays(GL_POINTS, 0, num_points)
    glFlush()


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
    glutInitDisplayMode(GLUT_RGBA)
    glutInitWindowSize(640, 480)
    glutCreateWindow("simple OpenGL example")

    init()
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)

    glutMainLoop()


if __name__ == "__main__":
    main()
