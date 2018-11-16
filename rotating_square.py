from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np
from framework import *


strFragmentShader = """
#version 330

out vec4 outputColor;
void main()
{
    outputColor = vec4(1.0, 0.0, 0.0, 1.0);
}
"""

theta = 0.0
num_points = 4
vertex_dim = 2

theta_loc = None


def init():
    vertices = np.array([[0, 1],
                         [1, 0],
                         [-1, 0],
                         [0, -1]],
                        dtype='float32').flatten()


    # Create the shaders and program
    shader_list = []
    shader_list.append(loadShader(GL_VERTEX_SHADER, "rotating.vert"))
    shader_list.append(createShader(GL_FRAGMENT_SHADER, strFragmentShader))
    program = createProgram(shader_list)
    for shader in shader_list:
        glDeleteShader(shader)

    glUseProgram(program)

    # Link theta to the shader
    global theta_loc
    theta_loc = glGetUniformLocation(program, "theta")
    glUniform1f(theta_loc, theta)


    # Setup the vertex buffer for storing vertex coordinates
    position_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, position_buffer)
    glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW)

    # Create a vertex array object
    glBindVertexArray(glGenVertexArrays(1))

    # Initialize the vertex position attribute from the vertex shader
    loc = glGetAttribLocation(program, "vPosition")
    glEnableVertexAttribArray(loc)
    glVertexAttribPointer(loc, vertex_dim, GL_FLOAT, GL_FALSE, 0, None)


def display():
    """Draw the points on the display."""
    glClearColor(0., 0., 0., 0.)
    glClear(GL_COLOR_BUFFER_BIT)
    global theta
    theta += 0.1
    glUniform1f(theta_loc, theta)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, num_points)
    glutSwapBuffers()       # Because we are using double buffering
    glutPostRedisplay()     # redisplay the window


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
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
    glutInitWindowSize(640, 480)
    glutCreateWindow("simple OpenGL example")

    init()
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)

    glutMainLoop()


if __name__ == "__main__":
    main()
