from cProfile import label
from configparser import Interpolation
import copy
import math
from timeit import repeat
from tkinter import simpledialog
from unicodedata import name
import numpy as np
import cv2
import sys
import os
import subprocess
import matplotlib.pyplot as plt
from constants import RESULT_DIR
import json

def saveArgument(ars, file):
    with open(file, 'w') as f:
        json.dump(ars.__dict__, f, indent=2)

def loadArgument(ars, file):
    with open(file, 'r') as f:
        ars.__dict__ = json.load(f)
    return ars

def checkpath(mypath):
    if not os.path.exists(mypath):
        os.makedirs(mypath)

def queryMicrofacetHistory(scenename, statefolder, iterations):
    rparams_stats_file = RESULT_DIR + "/{}/{}/".format(scenename, statefolder) + "rough.npy"
    aparams_stats_file = RESULT_DIR + "/{}/{}/".format(scenename, statefolder) + "albedo.npy"

    roughs = np.load(rparams_stats_file)
    albedo = np.load(aparams_stats_file)

    print("roughness ", roughs[int(iterations / 10)])
    print("albedo ", albedo[int(iterations / 10)])

def PlotStats(scenename, statefolder):
    params_stats_file = RESULT_DIR + "/{}/{}/".format(scenename, statefolder) + "parameters_rmse.npy"
    loss_stats_file = RESULT_DIR + "/{}/{}/".format(scenename, statefolder) + "loss.npy"
    params_rmse = np.load(params_stats_file)
    losses = np.load(loss_stats_file)

    
    xaxis = np.arange(losses.shape[0]) * 10
    xaxis2 = np.arange(params_rmse.shape[0]) * 10

    fig, axes = plt.subplots(2)
    fig.suptitle('Stats of Optimization (fix shape, opt albedo and sigma_t')
    axes[0].plot(xaxis, np.log10(np.abs(losses)), 'k', label="loss")
    axes[0].set_ylabel('log10 of loss')

    axes[1].plot(xaxis2, np.log10(params_rmse[:,0]), 'r', label="albedo")
    axes[1].plot(xaxis2, np.log10(params_rmse[:,1]), 'b', label="sigma_t")
    axes[1].set_xlabel('iterations')
    axes[1].set_ylabel('log10 of parameters\'s RMSE')
    axes[1].legend()
    
    plt.show()

def dumpPly(filename, vertices, faces):
    thefile = open('{}.ply'.format(filename), 'w')
    thefile.write("ply\n") 
    thefile.write("format ascii 1.0 \n")
    thefile.write("element vertex {}\n".format(vertices.shape[0]))
    # thefile.write("element  {}\n", vertices.shape[0])
    thefile.write("property float32 x\n")
    thefile.write("property float32 y\n")
    thefile.write("property float32 z\n")
    thefile.write("element face {}\n".format(faces.shape[0]))
    thefile.write("property list uint8 int32 vertex_indices\n")
    thefile.write("end_header\n")
    for i in range(vertices.shape[0]):
        thefile.write("{:04f} {:04f} {:04f}\n".format(vertices[i, 0],vertices[i, 1], vertices[i, 2]))
    for i in range(faces.shape[0]):
        thefile.write("3 {} {} {} \n".format(faces[i, 0], faces[i, 1], faces[i, 2]))

    thefile.close()

def getXYZ(phi, theta, radius):
    y = math.cos(theta) * radius
    x = math.sin(theta) * math.cos(phi) * radius
    z = math.sin(theta) * math.sin(phi) * radius
    return x,y,z


def generatecamerasAndLight(phiN, thetaN, radius, scene_name, light_name, radiuslight):
    phi_step = 360 / phiN
    theta_step = 180 / thetaN

    phi_degree = 0
    theta_degree = 0

    f = open("/home/dx/Research/psdr-cuda/examples/{}_sensor.xml".format(scene_name), "w")

    lightposes = np.zeros((0, 3), dtype=np.float32)

    for i in range(0, phiN):
        phi_degree += phi_step
        phi = math.radians(phi_degree)

        phi_l0 = math.radians(phi_degree + 90)
        phi_l1 = math.radians(phi_degree + 240)
        phi_l2 = math.radians(phi_degree - 20)
        
        for j in range(0, thetaN): 
            theta_degree += theta_step
            theta = math.radians(theta_degree)

            x,y,z = getXYZ(phi, theta, radius)
            text = """ <sensor type="perspective" id="sensor={}-{}-0">\n
            <string name="fov_axis" value="x" />
            <float name="fov" value="60" />
            <transform name="to_world">
                <lookat origin="{}, {}, {}" target="0.0, 0.0, 0.0" up="0, 1, 0" />
            </transform>
            </sensor> \n """.format(i, j, x, y, z, i, j)
            f.write(text)

            text = """ <sensor type="perspective" id="sensor={}-{}-1">\n
            <string name="fov_axis" value="x" />
            <float name="fov" value="60" />
            <transform name="to_world">
                <lookat origin="{}, {}, {}" target="0.0, 0.0, 0.0" up="0, 1, 0" />
            </transform>
            </sensor> \n """.format(i, j, x, y, z, i, j)
            f.write(text)

            text = """ <sensor type="perspective" id="sensor={}-{}-2">\n
            <string name="fov_axis" value="x" />
            <float name="fov" value="60" />
            <transform name="to_world">
                <lookat origin="{}, {}, {}" target="0.0, 0.0, 0.0" up="0, 1, 0" />
            </transform>
            </sensor> \n """.format(i, j, x, y, z, i, j)
            f.write(text)

            lightposes = np.concatenate((lightposes, [getXYZ(phi_l0, theta, radiuslight)]), dtype=np.float32)
            lightposes = np.concatenate((lightposes, [getXYZ(phi_l1, theta, radiuslight)]), dtype=np.float32)
            lightposes = np.concatenate((lightposes, [getXYZ(phi_l2, theta, radiuslight)]), dtype=np.float32)
    
    f.close()
    np.save("/home/dx/Research/psdr-cuda/examples/lights-sy-{}.npy".format(light_name), lightposes)


def generateRotateCamera(radius, scene_name, frames):
    phi_step = 0.1
    # theta_step = 0.1

    phi_degree = 0
    theta_degree = 90
    theta = math.radians(theta_degree)
    f = open("/home/dx/Research/psdr-cuda/examples/rotate_{}_sensor.xml".format(scene_name), "w")


    for i in range(0, frames):
        print(i)
        phi_degree += phi_step
        phi = math.radians(phi_degree)

        x,y,z = getXYZ(phi, theta, radius)
        text = """ <sensor type="perspective" id="sensor={}-0">\n
        <string name="fov_axis" value="x" />
        <float name="fov" value="60" />
        <transform name="to_world">
            <lookat origin="{}, {}, {}" target="{}, {}, {}" up="0, 1, 0" />
        </transform>
        </sensor> \n """.format(i, x, y, z, 0.3, 6.0, -0.5)
        f.write(text)
    f.close()


def generateLights(phiN, thetaN, radius, name):
    phi_step = 360 / phiN
    theta_step = 180 / thetaN

    phi_degree = 0
    theta_degree = 0

    lightposes = np.zeros((phiN * thetaN, 3), dtype=np.float32)

    for i in range(0, phiN):
        phi_degree = phi_degree % 360
        phi = math.radians(phi_degree)
        for j in range(0, thetaN): 
            theta_degree = theta_degree % 360
            theta = math.radians(theta_degree)

            y = math.cos(theta) * radius
            x = math.sin(theta) * math.cos(phi) * radius
            z = math.sin(theta) * math.sin(phi) * radius

            lightposes[i * thetaN + j, :] = [x, y, z]
            # print(x, y, z)
            theta_degree += theta_step
        phi_degree += phi_step
            
    np.save("/home/dx/Research/psdr-cuda/examples/lights-sy-{}.npy".format(name), lightposes)


def generatecamerasfromfile(file):
    lines = []
    with open(file) as cameralightinfo:
        lines = cameralightinfo.readlines()
        print(lines)

    f = open("/home/dx/Research/psdr-cuda/examples/essen-sensors-gantry.xml", "w")
    lights = np.zeros((0,3), dtype=np.float)

    for i in range(51):
        imageid = lines[i*6]
        print(imageid)
        cameraposition = lines[i*6+1]
        print(cameraposition)
        cameradirection = lines[i*6+2]
        cameradown = lines[i*6+3]
        cameraright = lines[i*6+4]
        lightposition = lines[i*6+5]

        positions = cameraposition.split()
        print("\n",positions)
        z = float(positions[2][1:-1]) / 10.0
        x = float(positions[3][:-1]) / 10.0
        y = float(positions[4][:-1]) / 10.0
        print(x, y, z)

        directions = cameradirection.split()
        print("\n",directions)
        dz = float(directions[2][1:-1])
        dx = float(directions[3][:-1])
        dy = float(directions[4][:-1])
        print(dx, dy, dz)

        right = cameraright.split()
        print("\n",right)
        rz = float(right[1][6:-1])
        rx = float(right[2][:-1])
        ry = float(right[3][:-1])
        print(rx, ry, rz)

        up = cameradown.split()
        print("\n",up)
        uz = -float(up[1][5:-1])
        ux = -float(up[2][:-1])
        uy = -float(up[3][:-1])
        print(ux, uy, uz)

        lightpos = lightposition.split()
        print("\n",lightpos)
        lz = float(lightpos[2][1:-2]) / 10.0
        lx = float(lightpos[3][:-2]) / 10.0
        ly = float(lightpos[4][:-2]) / 10.0
        print(lx, ly, lz)

        lp = np.array([lx, ly, lz])
        lights = np.concatenate((lights, [lp]), axis=0)

        tx = x + dx
        ty = y + dy
        tz = z + dz

        text = """ <sensor type="perspective" id="sensor-{}">
            <string name="fov_axis" value="x" />
            <float name="fov" value="10" />
            <transform name="to_world">
                <lookat origin="{}, {}, {}" target="{}, {}, {}" up="{}, {}, {}" />
            </transform>
            </sensor> \n """.format(imageid, x, y, z, tx, ty, tz, ux, uy, uz)
        f.write(text)

    np.save("/home/dx/Research/psdr-cuda/examples/essen-lights-gantry.npy", lights)


def convertTexture(source, target):
    texture = cv2.imread(source, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    tex = texture.astype(np.float32)
    tex = tex / 255.0
    tex[tex < 0.003] = 0.003
    tex[tex > 0.995] = 0.995
    cv2.imwrite(target, tex)
    print(tex)

def generateInitial(source, target):
    texture = cv2.imread(source, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    tex = texture.astype(np.float32)
    tex = tex / 255.0 * 0.0 + 0.5
    cv2.imwrite(target, tex)
    print(tex.shape)
    # print(tex)

def generateInitTexture(filedir, resx, resy):
    texture = np.zeros((resx, resy, 3), dtype=np.float32)
    texture = texture + 0.5
    cv2.imwrite(filedir, texture)


def generateTexture(filedir, resx, resy):
    step = int(resx / 2)
    sigma_t_left  = np.zeros((step, resy, 3), dtype=np.float32)
    sigma_t_right = np.zeros((step, resy, 3),  dtype=np.float32)

    sigma_t_left[:,:,0] = 0.1
    sigma_t_left[:,:,1] = 0.1
    sigma_t_left[:,:,2] = 0.1

    sigma_t_right[:,:,0] = 1.99
    sigma_t_right[:,:,1] = 1.99  
    sigma_t_right[:,:,2] = 1.99  


    sigma_image = np.concatenate((sigma_t_left, sigma_t_right), axis=0)
    cv2.imwrite(filedir + "/sigma_t.exr", sigma_image)

    albedo_t_left  = np.zeros((step, resy, 3),  dtype=np.float32)
    albedo_t_right = np.zeros((step, resy, 3),  dtype=np.float32)

    albedo_t_left[:,:,0] = 0.4
    albedo_t_left[:,:,1] = 0.4
    albedo_t_left[:,:,2] = 0.5

    albedo_t_right[:,:,0] = 0.9
    albedo_t_right[:,:,1] = 0.7  
    albedo_t_right[:,:,2] = 0.7  


    albedo_image = np.concatenate((albedo_t_left, albedo_t_right), axis=0)
    cv2.imwrite(filedir + "/albedo.exr", albedo_image)

def remesh(input_dir, output_dir, copy_dir, vertices_count):
    cmd = ['/bin/bash', '-i', '-c', 'imesh -p 6 -r 2 -o {} -v {} {}'.format(output_dir, vertices_count, input_dir)]
    subprocess.call(cmd)
    cmd2 = ['/bin/bash', '-i', '-c', 'cp', '{}'.format(output_dir), '{}'.format(copy_dir)]
    subprocess.call(cmd2)

from constants import TEXTURE_DIR, RAW_TEXTURE_DIR

def generateTextureFromImage(name, repeat):
    source = RAW_TEXTURE_DIR+"/{}.jpg".format(name)
    target = TEXTURE_DIR+"/{}".format(name)
    print(source)
    texture = cv2.imread(source, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    texture = cv2.cvtColor(texture, cv2.COLOR_RGB2BGR)
    texture = cv2.resize(texture, (1024, 1024))
    v = []
    for i in range(0, repeat):
        h = []
        for j in range(0, repeat):
            h.append(texture)
        repeatTexture = np.concatenate(h, axis=0)
        v.append(repeatTexture)
    texture = np.concatenate(v, axis=1)
    texture = cv2.resize(texture, (1024, 1024))

    
    tex = texture.astype(np.float32)

    tex = tex / 255.0
    tex[tex < 0.003] = 0.003
    tex[tex > 0.995] = 0.995
    
    alebdo_tex = 0.1 + 0.895 * tex
    print(alebdo_tex.shape)
    alebdo_tex = cv2.cvtColor(alebdo_tex, cv2.COLOR_RGB2BGR)
    # dark place has high sigma_t
    # tex = 1.0 - tex
    ttt = np.mean(tex, axis=2, keepdims=True)
    sigma_tex = (ttt - np.min(ttt)) / (np.max(ttt) - np.min(ttt))
    sigma_tex = sigma_tex * sigma_tex * 3.0

    sig_text = np.concatenate((sigma_tex, sigma_tex, sigma_tex), axis=2)
    sig_text[sig_text<0.3] = 0.3
    sig_text = cv2.cvtColor(sig_text, cv2.COLOR_RGB2BGR)
    # tx = np.std(tex, axis=2)
    # sigma_tex = tex * 0.0
    # print(tx[tx<0.1])
    # exit(0)
    # maxk1 = (tx < 0.05)
    # mask2 = (tx > 0.05)
    # tx[maxk1] = tx[maxk1] * 100.0
    # tx[mask2] = tx[mask2] * 2.0
    
    # sigma_tex[:,:,0] = tx[:,:]
    # sigma_tex[:,:,1] = tx[:,:]
    # sigma_tex[:,:,2] = tx[:,:]

    cv2.imwrite(target + "_albedo_texture.exr", alebdo_tex)
    cv2.imwrite(target + "_sigma_texture.exr", sig_text)

    albedo_init_tex = np.zeros(tex.shape, dtype=tex.dtype) + 0.6
    sigma_init_tex = np.zeros(tex.shape, dtype=tex.dtype) + 0.3
    relectance_init_tex = np.zeros(tex.shape, dtype=tex.dtype) + 0.11
    cv2.imwrite(target + "_albedo_init_texture.exr", albedo_init_tex)
    cv2.imwrite(target + "_sigma_init_texture.exr", sigma_init_tex)
    cv2.imwrite(target + "_reflectance_init_texture.exr", relectance_init_tex)


# if __name__ == "__main__":
    # generateRotateCamera(20, "gantry", 3600)
    # generateRotateCamera(100, dragon, 3600)
    # generatecamerasAndLight(4, 4, 11, "bunny", "bunny", 45)
    # generatecameras(6, 6, 12)
    # PlotStats("cube_bump", "joint_sss_temp_0")
    # generateTextureFromImage("yellow", 4)
    # generateTextureFromImage("yellow")
    # remesh('/home/dx/Research/psdr-cuda/results/bunny_small/temp2/obj_8000.ply', '/home/dx/Research/psdr-cuda/results/bunny_small/mesh.obj', '/home/dx/Research/psdr-cuda/results/bunny_small/mesh_{}.obj'.format(args.seed), 300)
    # generatecameras(6, 6, 9)
    # generateLights(6, 6, 45, "platigon")
    # generateLights(5, 4, 45, "cube")

    # generatecameras(6, 6, 5)

    # generateLights(5, 4, 45, "cube")

# def generateVideoforOpt(step, file):  
#     # name = file + "/iter_{}.exr".format(i * step)
#     for i in range(iterations):
#         name = file + "/iter_{}.exr".format(i * step)
#         cv2.imread()

# generateInitTexture('/home/dx/Research/psdr-cuda/results/kiwi/texture.exr', 1024, 1024)
    # generatecamerasfromfile('/home/dx/Downloads/calibration.txt')

# generate camera positions an camera positoins


    # queryMicrofacetHistory("seal_1_sparse", "joint_bsdf_temp_1", 2500)
# generateTexture("/home/dx/Research/psdr-cuda/examples/textures/",10,10)
# convertTexture("/home/dx/Research/psdr-cuda/examples/textures/orange_slice.jpeg", "/home/dx/Research/psdr-cuda/examples/textures/orange_slice.exr")
# convertTexture("/home/dx/Research/psdr-cuda/examples/textures/kiwi.jpg", "/home/dx/Research/psdr-cuda/examples/textures/kiwi.exr")
# convertTexture("/home/dx/Research/psdr-cuda/examples/textures/watermelon.png", "/home/dx/Research/psdr-cuda/examples/textures/watermelon.exr")
# convertTexture("/home/dx/Research/psdr-cuda/examples/textures/dragonfruit.jpg", "/home/dx/Research/psdr-cuda/examples/textures/dragonfruit.exr")
# generateInitial("/home/dx/Research/psdr-cuda/examples/textures/dragonfruit.jpg", "/home/dx/Research/psdr-cuda/examples/textures/dragonfruit_init.exr")

    # lights = np.zeros((4, 3), dtype=np.float32)
    # lights[0, :] = [0.0, 5.2, 0.152505]
    # lights[1, :] = [0.0, -5.2, 0.152505]
    # lights[2, :] = [0.0, -5.2, 0.152505]
    # lights[3, :] = [0.0, 5.2, 0.152505]
    # np.save("/home/dx/Research/psdr-cuda/examples/lights-sy-layer.npy", lights)