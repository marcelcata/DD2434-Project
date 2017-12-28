#Efficient Computation of SSK

#include <shogun/kernel/string/SubsequenceStringKernel.h>
#include <shogun/kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <shogun/features/StringFeatures.h>

#using namespace shogun;

#Compute kernel for strings s and t
def compute(s,t):
	len_s = len(s)
	len_t = len(t)
	max_len= 5 #max lenght
	m_lambda = 0.5 #weights
	Kp=[] # kp = k'
	Kpp=[] # Kpp = k''
	

	# allocating memory for computing K' (Kp)  ->> ??
	# initialize for 0 subsequence length for both the strings
	for i in range(len_s):
		for j in range(len_t):
			Kp[0][i][j]= 1.0
			
	# computing of the K' (Kp) function using equations
	# shown in Lodhi et. al. See the class documentation for
	# definitions of Kp and Kpp
	for i in range(max_len):
		for j in range(len_s - 1):
			Kpp=0.0
			for x in range(len_t - 1):
				Kpp=m_lambda*(Kpp+m_lambda*(s[j]==t[x])*Kp[i][j][x])
				Kp[i+1][j+1][x+1]=m_lambda*Kp[i+1][j][x+1]+Kpp

	# compute the kernel function
	K=0.0;
	for i in range(max_len):
		for j in range(len_s):
			for x in range(len_t):
				K+=m_lambda*m_lambda*(s[j]==t[x])*Kp[i][j][x]

	# cleanup  ->> ??
	return K;
