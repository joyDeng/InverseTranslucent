from tkinter import W
from turtle import width
from numpy.core.numeric import indices
import torch 
import torch.nn as nn 
import numpy as np
import cv2
import scipy
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import factorized
import random

def active_sensors(batch, num_sensors):
    indices = torch.tensor(random.sample(range(num_sensors), batch))
    return indices

def active_light(batch, num_lights):
    indices = torch.tensor(random.sample(range(num_lights), batch))
    return indices

def texture_range_loss(A, S, weight):
	lossS = torch.pow(torch.where(S < 0, -S, torch.where(S > 100.0, S - 100.0, torch.zeros_like(S))), 2)
	lossA = torch.pow(torch.where(A < 0.0, -A, torch.where(A > 1.0, A - 1, torch.zeros_like(A))), 2)
	loss = (lossA.mean() + lossS.mean()) * weight
	return loss

def ImageLaplacianUniform(width):
	with torch.no_grad():
		W = width * width
		idx = torch.arange(W,)
		# , use torch.div(a, b, rounding_mode='floor').
		ridx = torch.div(torch.arange(W, dtype=torch.int), width, rounding_mode="floor")
		cidx = torch.arange(W, dtype=torch.int) % width
		
		# print("idx=", idx)
		# print("ridx=", ridx)
		# print("cidx=", cidx)
		# print("nidx")
		nidx = torch.zeros((W, 4))
		nidx[:, 0] = (ridx - 1) * width + cidx
		nidx[:, 1] = (ridx) * width + cidx - 1
		nidx[:, 2] = (ridx) * width + cidx + 1
		nidx[:, 3] = (ridx + 1) * width + cidx
		upzero = torch.arange(width)
		downzero = torch.arange(width) + width * (width - 1)
		leftzero = torch.arange(width) * width  
		rightzero = torch.arange(width) * width + width - 1
		# print("upzero = ", upzero)
		# print("downzero = ", downzero)
		# print("rightzero = ", rightzero)
		# print("leftzero = ", leftzero)
		nidx[leftzero, 1] = -1
		nidx[rightzero, 2] = -1
		nidx[upzero, 0] = -1
		nidx[downzero, 3] = -1
		# print("nidx", nidx)
		midx = torch.zeros((W, 4),)
		midx[:, 0] = idx
		midx[:, 1] = idx
		midx[:, 2] = idx
		midx[:, 3] = idx
		# print("midx", midx)
		ii = midx.flatten()
		jj = nidx.flatten()
		adj = torch.stack([ii[jj >= 0], jj[jj >=0]])
		# print("adj", adj)
		adjvalues = torch.ones(adj.shape[1], dtype=torch.float)
		# exit(0)
		diag_idx = adj[0]
		lidx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
		values = torch.cat((-adjvalues, adjvalues))
	# print("values", values)
	# print(adj[0])
	
	return torch.sparse_coo_tensor(lidx.to(torch.device("cuda")), values.to(torch.device("cuda")), (W, W)).coalesce()
	# print(L)


def compute_image_matrix(width, lambda_):
	L = ImageLaplacianUniform(width)
	W = width * width
	idx = torch.arange(width * width, dtype=torch.long, device="cuda")
	eye = torch.sparse_coo_tensor(torch.stack((idx, idx), dim=0), torch.ones(W, dtype=torch.float, device='cuda'), (W, W))
	AM = torch.add(eye, lambda_*L)
	return AM.coalesce()

def compute_two_image_matrix(width, lambda1_, lambda2_):
	L = ImageLaplacianUniform(width)
	W = width * width
	idx = torch.arange(width * width, dtype=torch.long, device="cuda")
	eye = torch.sparse_coo_tensor(torch.stack((idx, idx), dim=0), torch.ones(W, dtype=torch.float, device='cuda'), (W, W))
	AM = torch.add(eye, lambda1_*L)
	SM = torch.add(eye, lambda2_*L)
	return AM.coalesce(), SM.coalesce()

def mesh_uni_laplacian(verts, edges, laplacian=0.5):
	with torch.no_grad():
		V = verts.shape[0]
		# print("edges shape", edges.shape)
		# print(edges)
		e0, e1 = edges.unbind(1)


		idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
		idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
		idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

		# First, we construct the adjacency matrix,
		# i.e. A[i, j] = 1 if (i,j) is an edge, or
		# A[e0, e1] = 1 &  A[e1, e0] = 1
		ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
		A = torch.sparse.FloatTensor(idx, ones, (V, V))

		# the sum of i-th row of A gives the degree of the i-th vertex
		deg = torch.sparse.sum(A, dim=1).to_dense()

		# We construct the Laplacian matrix by adding the non diagonal values
		# i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
		deg0 = deg[e0]
		deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
		deg1 = deg[e1]
		deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
		val = torch.cat([deg0, deg1])
		L = torch.sparse.FloatTensor(idx, val, (V, V))

		# Then we add the diagonal values L[i, i] = -1.
		idx = torch.arange(V, device=verts.device)
		idx = torch.stack([idx, idx], dim=0)
		ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
		L = torch.sparse.FloatTensor(idx, ones, (V, V)) - L

		T = torch.sparse.FloatTensor(idx, ones, (V, V)) + laplacian * L

		J = csc_matrix((T.cpu().detach().coalesce().values(), T.cpu().detach().coalesce().indices()), shape=(V, V))

		J_solver = factorized(J)

		return J, J_solver



def mesh_cot_laplacian(verts, faces, lambda1=0.5):
	# compute L once per mesh subdiv. 
	with torch.no_grad():
		V, F = verts.shape[0], faces.shape[0]
		face_verts = verts[faces]
		v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
		# Side lengths of each triangle, of shape (sum(F_n),)
		# A is the side opposite v1, B is opposite v2, and C is opposite v3
		A = (v1 - v2).norm(dim=1)
		B = (v0 - v2).norm(dim=1)
		C = (v0 - v1).norm(dim=1)
		# Area of each triangle (with Heron's formula); shape is (sum(F_n),)
		s = 0.5 * (A + B + C)
		# note that the area can be negative (close to 0) causing nans after sqrt()
		# we clip it to a small positive value
		area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-6).sqrt()
		# Compute cotangents of angles, of shape (sum(F_n), 3)
		A2, B2, C2 = A * A, B * B, C * C
		cota = (B2 + C2 - A2) / area
		cotb = (A2 + C2 - B2) / area
		cotc = (A2 + B2 - C2) / area
		cot = torch.stack([cota, cotb, cotc], dim=1)
		cot /= 4.0
		# Construct a sparse matrix by basically doing:
		# L[v1, v2] = cota
		# L[v2, v0] = cotb
		# L[v0, v1] = cotc
		ii = faces[:, [1, 2, 0]]
		jj = faces[:, [2, 0, 1]]
		idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
		L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))
		L += L.t() 
		norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
		i_norm_w = 1.0 / norm_w
		idx1 = torch.stack([torch.arange(V, dtype=ii.dtype, device=ii.get_device()), torch.arange(V, dtype=jj.dtype, device=jj.get_device())])
		W = torch.sparse.FloatTensor(idx1, norm_w.view(-1), (V, V))
		W2 = torch.sparse.FloatTensor(idx1, i_norm_w.view(-1), (V, V))
		# print("W", W.to_dense())
		L -= W
		T =torch.sparse.mm(W2, L)
		# print("L", T.to_dense())
	# print(idx1.shape)
	# print(idx.shape)
	# print(idx)
	# print(norm_w.shape)
	# print(norm_w)
	# exit(0)


	# J = torch.eye(V, device=torch.device('cuda:0')) + lambda1 * L.to_dense()
	
	# norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
	# idx = norm_w > 1e-6
	# norm_w[idx] = 1.0 / norm_w[idx]
	# data = cot.view(-1).cpu()
	# print(data.shape, idx.cpu().shape)
	J = lambda1 * csc_matrix((T.cpu().detach().coalesce().values(), T.cpu().detach().coalesce().indices()), shape=(V, V)) + eye(V)
	# print(IL)
	
	# print(J)
	J_solver = factorized(J)
	# exit(0)
		

	# exit(0)
		
		# J_i = J.inverse()
		# print("J", J.shape, J)
		# print("J_i", J_i.shape, J_i)
	# loss = L.mm(verts) * norm_w - verts
	return J, J_solver



def mesh_laplacian_smoothing(verts, faces, weight):

	# compute L once per mesh subdiv. 
	with torch.no_grad():
		V, F = verts.shape[0], faces.shape[0]
		face_verts = verts[faces]
		v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
		# Side lengths of each triangle, of shape (sum(F_n),)
		# A is the side opposite v1, B is opposite v2, and C is opposite v3
		A = (v1 - v2).norm(dim=1)
		B = (v0 - v2).norm(dim=1)
		C = (v0 - v1).norm(dim=1)
		# Area of each triangle (with Heron's formula); shape is (sum(F_n),)
		s = 0.5 * (A + B + C)
		# note that the area can be negative (close to 0) causing nans after sqrt()
		# we clip it to a small positive value
		area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-6).sqrt()
		# Compute cotangents of angles, of shape (sum(F_n), 3)
		A2, B2, C2 = A * A, B * B, C * C
		cota = (B2 + C2 - A2) / area
		cotb = (A2 + C2 - B2) / area
		cotc = (A2 + B2 - C2) / area
		cot = torch.stack([cota, cotb, cotc], dim=1)
		cot /= 4.0
		# Construct a sparse matrix by basically doing:
		# L[v1, v2] = cota
		# L[v2, v0] = cotb
		# L[v0, v1] = cotc
		ii = faces[:, [1, 2, 0]]
		jj = faces[:, [2, 0, 1]]
		idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
		L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))
		# Make it symmetric; this means we are also setting
		# L[v2, v1] = cota
		# L[v0, v2] = cotb
		# L[v1, v0] = cotc
		L += L.t() 
		norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
		idx = norm_w > 1e-6
		norm_w[idx] = 1.0 / norm_w[idx]

	loss = L.mm(verts) * norm_w - verts
	loss = loss.norm(dim=1)

	return loss.mean() * weight
 
def mesh_normal_consistency(verts, faces, edges, weight): 
	fa, fb = edges[:, 2], edges[:, 3] # two faces (index) sharing one edge a <--> b
	va0, va1, va2 = verts[faces[fa, 0]], verts[faces[fa, 1]], verts[faces[fa, 2]]
	vb0, vb1, vb2 = verts[faces[fb, 0]], verts[faces[fb, 1]], verts[faces[fb, 2]]
	n0 = (va1 - va0).cross(va2 - va0, dim=1)  
	n1 = (vb1 - vb0).cross(vb2 - vb0, dim=1) 
	n0 = n0 / (torch.norm(n0) + 1e-6)
	n1 = n1 / (torch.norm(n1) + 1e-6)
	loss = 1 - torch.cosine_similarity(n0, n1, dim=1)
	
	return loss.mean() * weight

def mesh_edge_loss(verts, edges, weight, target_length: float = 0.0):
	va, vb = edges[:, 0], edges[:, 1] # two verts (index) sharing one edge a <--> b 
	v0, v1 = verts[va, :], verts[vb, :]
	loss = ((v0 - v1).norm(dim=1, p=2) - target_length) ** 2.0
	
	return loss.mean() * weight

if __name__ == "__main__":
	ImageLaplacianUniform(16)