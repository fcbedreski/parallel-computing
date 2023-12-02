int main() {

    for(i=1;i<(n-1);i++)
	{
		T[i]=0;

	}

	for(j=0;j<m;j++)
	{

		/* - Assignment comments -
		Vectorization: 'simd' (single instruction multiple data) directive
		Operates on more than one element per iteration
		It need to be in this loop, because in the outer loop it would apply 'simd' for each loop in 'i',
		for each thread, changing the same value several times.
		*/
		#pragma omp simd 
		for(i=1;i<n-1;i++)
		{
			T_new[i]=(((Hg*dX*dX)/(2*K))+((T[i-1]+T[i+1])/2));

		}
        T_new[n-1]=(((K*T[n-2])/dX)+(h*T_amb)+(Hg*(dX/2)))* (dX/(K+(dX*h)));

		swap(T, T_new)
	}

    return 0; 

}
