from OpenGL.GL import *
import os
import sys


def createShader(shader_type, shader_file):
    """Create and compile a shader from a string."""
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


def loadShader(shaderType, shaderFile):
    """Load a shader from a file."""
    # check if file exists, get full path name
    strFilename = findFileOrThrow(shaderFile)
    shaderData = None
    with open(strFilename, 'r') as f:
        shaderData = f.read()

    shader = glCreateShader(shaderType)
    glShaderSource(shader, shaderData)
    glCompileShader(shader)

    status = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if status == GL_FALSE:
        strInfoLog = glGetShaderInforLog(shader)
        strShaderType = ""
        if shaderType is GL_VERTEX_SHADER:
            strShaderType = "vertex"
        elif shaderType is GL_GEOMETRY_SHADER:
            strShaderType = "geometry"
        elif shaderType is GL_FRAGMENT_SHADER:
            strShaderType = "fragment"

        print "Compilation failure for " + strShaderType + " shader:\n" + strInfoLog

    return shader


def findFileOrThrow(strBasename):
    """Looks for the shader file, or throws an error if it is not found."""
    local_dir = "shaders" + os.sep
    global_dir = ".." + os.sep + "shaders" + os.sep

    strFilename = local_dir + strBasename
    if os.path.isfile(strFilename):
        return strFilename

    strFilename = global_dir + strBasename
    if os.path.isfile(strFilename):
        return strFilename

    raise IOError('Could not find target file ' + strBasename)


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
