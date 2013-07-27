////////////////////////////////////////////////////////////////////////////////////////////////////
//  gpRepel : An R package for GPU computing
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; version 3 of the License.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
//
//  Author: Piotr Wąsiewicz pwasiewi@gmail.com
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <R.h>
#include "gpRepel.h"

typedef thrust::tuple<Numeric,Numeric> 								Numeric2;
typedef typename thrust::device_vector<Numeric>::iterator         	NumericIterator;
typedef typename thrust::tuple<NumericIterator, NumericIterator>  	NumericIteratorTuple;
typedef typename thrust::zip_iterator<NumericIteratorTuple>       	Numeric2Iterator;
typedef thrust::tuple<Numeric,Numeric,Numeric> 						Numeric3;
typedef typename thrust::tuple<NumericIterator, NumericIterator, NumericIterator>  NumericIteratorTuple3;
typedef typename thrust::zip_iterator<NumericIteratorTuple3>       	Numeric3Iterator;

const int NAVG=80;

////////////////////////////////////////////////////////////////////////////////////////////////////
// VecReorder - the gpu functor implementing the dot product between 3d vectors
//
struct VecReorder : public thrust::binary_function<Numeric2,Numeric2,Numeric>
{
    Numeric w, maxb;
    Numeric ai,bi,av,bv, result;

    __host__ __device__
    VecReorder(Numeric w, Numeric maxb) : w(w), maxb(maxb) {}
    __host__ __device__
        Numeric operator()(const Numeric2& a, const Numeric2& b) const
        {
            Numeric ai=(int) thrust::get<0>(a) % (int) maxb;
            Numeric av=thrust::get<1>(a);
            Numeric bi=(int) thrust::get<0>(b) % (int) maxb;
            Numeric bv=thrust::get<1>(b);
	    	int lastone = (int) maxb*((int) thrust::get<0>(a) / (int) maxb)-1;
	    	if(ai > bi)
            	return thrust::get<0>(a);
	    	else
	    		return lastone;
        }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// minus_and_divide_zip - the gpu functor implementing moving average in a point 
//
struct  minus_and_divide_zip : public thrust::binary_function<Numeric3,Numeric3,Numeric>
{
    Numeric w, maxb;
    Numeric ai,bi,av,bv, result;

    __host__ __device__
    minus_and_divide_zip(Numeric w, Numeric maxb) : w(w), maxb(maxb) {}
    __host__ __device__
    Numeric operator()(const Numeric3& a, const Numeric3& b) const
    {
		Numeric ai=(int) thrust::get<0>(a) % (int) maxb;
		Numeric av=thrust::get<1>(a);
		Numeric bi=(int) thrust::get<0>(b) % (int) maxb;
		Numeric bv=thrust::get<1>(b);
	    int lastone = (int) maxb*((int) thrust::get<0>(a) / (int) maxb);
	    if(ai > bi)
                return (av - bv)/w;
	    else
	    	if((int)w - 1 != (int) ai)
	    		return (thrust::get<2>(b) - thrust::get<1>(b)) / ((int)w - (int)ai -1);
	    	else
	    		return 0;
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
// simple_moving_average - GPU function of the simple average with a window w points forward, 
// after a given point; idata, vout - input and output matrices with m (rows) x n (cols) dimensions,
// 
template <typename InputVector, typename OutputVector>
void simple_moving_average(size_t m, size_t n, const InputVector& idata, size_t w, OutputVector& vout)
{
    typedef typename InputVector::value_type T;

    if (idata.size() < w)
        return;
    thrust::device_vector<size_t> output(m*n);
    thrust::device_vector<Numeric> voutput(m*n);
    thrust::device_vector<Numeric> data(m*n);
    thrust::device_vector<Numeric> vindex(m*n);
    thrust::sequence(vindex.begin(),vindex.end());

    thrust::inclusive_scan(idata.begin(), idata.end(), data.begin());

    Numeric2Iterator first = thrust::make_zip_iterator(thrust::make_tuple(vindex.begin(), data.begin()));
    Numeric2Iterator firstw = thrust::make_zip_iterator(thrust::make_tuple(vindex.begin() + w, data.begin() + w));
    Numeric2Iterator last  = thrust::make_zip_iterator(thrust::make_tuple(vindex.end(),   data.end()));

    thrust::transform(firstw, last, first, output.begin(), VecReorder(w,m));
 
    thrust::gather(output.begin(), output.end(), data.begin(), voutput.begin());

    Numeric3Iterator first3 = thrust::make_zip_iterator(thrust::make_tuple(vindex.begin(), data.begin(), voutput.begin()));
    Numeric3Iterator firstw3 = thrust::make_zip_iterator(thrust::make_tuple(vindex.begin() + w, data.begin() + w, voutput.begin() + w));
    Numeric3Iterator last3  = thrust::make_zip_iterator(thrust::make_tuple(vindex.end(), data.end(), voutput.end()));

    thrust::transform(firstw3, last3, first3, vout.begin(), minus_and_divide_zip(w,m));
    thrust::fill(vout.end()-w,vout.end(),vout[vout.size()-w-1]);
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// gprpostmave - host-gpu function for the gpu simple average function simple_moving_average 
// with a window w points forward, after a given point
// pint, pout - input and output matrices with a (rows) x b (cols) dimensions
//
void gprpostmave(PNumeric pint, PInteger a, PInteger b, PInteger win, PNumeric pout) {

    // window size of the moving average
    size_t w = win[0];
    size_t m = a[0];//row number
    size_t n = b[0];//column number

    // transfer data to the device
    thrust::device_vector<Numeric> gveca(a[0]*b[0]);
    thrust::copy(pint,pint+a[0]*b[0],gveca.begin());

    thrust::device_vector<Numeric> gvecb(a[0]*b[0]);

    simple_moving_average(m,n,gveca, w, gvecb);

    // transfer data back to host
    thrust::copy(gvecb.begin(), gvecb.end(), pout);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// print an array m x n with vectors in columns
//
template <typename T>
void print(size_t m, size_t n, thrust::device_vector<T>& d_data)
{
    thrust::host_vector<T> h_data = d_data;

    for(size_t i = 0; i < n; i++)
    {
        for(size_t j = 0; j < m; j++)
            std::cout << " " << h_data[j + i * m] << " ";
        std::cout << "\n";
    }
    std::cout << "\n";
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// print an array m x n with vectors in columns as one vector: one after another
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
void printvec(size_t m, size_t n, thrust::device_vector<T>& d_data)
{
    thrust::host_vector<T> h_data = d_data;

    for(size_t i = 0; i < m*n; i++)
        //std::cout << fixed << setw(5) << setprecision(2) << h_data[i] << " ";
        std::cout << " " << h_data[i] << " ";
    std::cout << "\n";
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// gprpremave - host-gpu function for the gpu simple average function simple_moving_average 
// with a window w points back, before a given point
// pint, pout - input and output matrices with a (rows) x b (cols) dimensions
//
void gprpremave(PNumeric pint, PInteger a, PInteger b, PInteger win, PNumeric pout) {

    // window size of the moving average
    size_t w = win[0];
    size_t m = a[0];//row number
    size_t n = b[0];//column number

    // transfer data to the device
    thrust::device_vector<Numeric> gveca(a[0]*b[0]);
    thrust::copy(pint,pint+a[0]*b[0],gveca.begin());

    thrust::device_vector<Numeric> gvecb(a[0]*b[0]);

    thrust::reverse(gveca.begin(), gveca.end());
    simple_moving_average(m,n,gveca, w, gvecb);
    thrust::reverse(gvecb.begin(), gvecb.end());
    thrust::reverse(gveca.begin(), gveca.end());

    // transfer data back to host
    thrust::copy(gvecb.begin(), gvecb.end(), pout);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// plus_and_divide - a gpu functor of two arguments added and divided by a constant
//
template <typename T>
struct plus_and_divide : public thrust::binary_function<T,T,T>
{
    T w;

    __host__ __device__
    plus_and_divide(T w) : w(w) {}

    __host__ __device__
    T operator()(const T& a, const T& b) const
    {
        return (a + b) / w;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// double_moving_average - gpu function for the gpu average of two moving averages with windows 
// after and before the given point, joint of two simple_moving_average (one on a reversed copy)
template <typename InputVector, typename OutputVector>
void double_moving_average(size_t m, size_t n, const InputVector& igva, size_t w, OutputVector& gvd)
{
    typedef typename InputVector::value_type T;
    if (igva.size() < w)
        return;

    thrust::device_vector<T> gva(igva.size());
    thrust::device_vector<T> gvb(igva.size());
    thrust::device_vector<T> gvc(igva.size());
    thrust::copy(igva.begin(), igva.end(), gva.begin());

    simple_moving_average(m,n,gva, w, gvb);
    thrust::reverse(gva.begin(), gva.end());
    simple_moving_average(m,n,gva, w, gvc);
    thrust::reverse(gvc.begin(), gvc.end());
    thrust::reverse(gva.begin(), gva.end());
    thrust::transform(gvc.begin(), gvc.end(), gvb.begin(), gvd.begin(), plus_and_divide<T>(T(2)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// gprpremave - host-gpu function for the gpu average of two moving averages with windows 
// after and before the given point, joint of gprpremave and gprpostmave
// pint, pout - input and output matrices with a (rows) x b (cols) dimensions
//
void gprmoverage(PNumeric pint, PInteger a, PInteger b, PInteger win, PNumeric pout) {

    // window size of the moving average
    size_t w = win[0];
    size_t m = a[0];//row number
    size_t n = b[0];//column number

    // transfer data to the device
    thrust::device_vector<Numeric> gveca(a[0]*b[0]);
    thrust::copy(pint,pint+a[0]*b[0],gveca.begin());

    thrust::device_vector<Numeric> gvecz(a[0]*b[0]);
    double_moving_average(m,n,gveca,w,gvecz);

    // transfer data back to host
    thrust::copy(gvecz.begin(), gvecz.end(), pout);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// gprbasavoff - host-gpu function for a wave average with a small window divided by its baseline 
// (double average with a large window)
//
void gprbasavoff(PNumeric pint, PInteger a, PInteger b, PInteger win1, PInteger win2, PNumeric pout) {

    // window size of the moving average
    size_t w1 = win1[0];//smaller window
    size_t w2 = win2[0];//larger window
    size_t m = a[0];//row number
    size_t n = b[0];//column number

    // transfer data to the device
    thrust::device_vector<Numeric> gveca(a[0]*b[0]);
    thrust::copy(pint,pint+a[0]*b[0],gveca.begin());

    thrust::device_vector<Numeric> gvecb(a[0]*b[0]);
    thrust::device_vector<Numeric> gvecc(a[0]*b[0]);
    thrust::device_vector<Numeric> gvecd(a[0]*b[0]);

    double_moving_average(m,n,gveca, w1, gvecb);
    double_moving_average(m,n,gveca, w2, gvecc);

    thrust::transform(gvecb.begin(), gvecb.end(), gvecc.begin(), gvecd.begin(), thrust::divides<Numeric>());

    // transfer data back to host
    thrust::copy(gvecd.begin(), gvecd.end(), pout);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// gprbasoroff - host-gpu function for a wave divided by its baseline (double average 
// with a large window)
//
void gprbasoroff(PNumeric pint, PInteger a, PInteger b, PInteger win1, PNumeric pout) {

    // window size of the moving average
    size_t w1 = win1[0];//smaller window

    size_t m = a[0];//row number
    size_t n = b[0];//column number

    // transfer data to the device
    thrust::device_vector<Numeric> gveca(a[0]*b[0]);
    thrust::copy(pint,pint+a[0]*b[0],gveca.begin());

    thrust::device_vector<Numeric> gvecb(a[0]*b[0]);

    double_moving_average(m,n,gveca, w1, gvecb);

    thrust::transform(gveca.begin(), gveca.end(), gvecb.begin(), gveca.begin(), thrust::divides<Numeric>());

    // transfer data back to host
    thrust::copy(gveca.begin(), gveca.end(), pout);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// gprdiff - host-gpu function for shifted by the w window one wave copies substraction
//
void gprdiff(PNumeric pint, PInteger a, PInteger b, PInteger win1, PNumeric pout) {
    size_t w = win1[0];//difference window

    // transfer data to the device
    thrust::device_vector<Numeric> gveca(a[0]*b[0]);
    thrust::copy(pint,pint+a[0]*b[0],gveca.begin());

    thrust::device_vector<Numeric> gvecb(a[0]*b[0]);
    thrust::transform(gveca.begin()+w, gveca.end(), gveca.begin(), gvecb.begin(), thrust::minus<Numeric>());

    // transfer data back to host
    thrust::copy(gvecb.begin(), gvecb.end(), pout);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// gprdiffrev - host-gpu function for shifted by the w window one wave reversed copies substraction
//
void gprdiffrev(PNumeric pint, PInteger a, PInteger b, PInteger win1, PNumeric pout) {
    size_t w = win1[0];//difference window

    // transfer data to the device
    thrust::device_vector<Numeric> gveca(a[0]*b[0]);
    thrust::copy(pint,pint+a[0]*b[0],gveca.begin());

    thrust::device_vector<Numeric> gvecb(a[0]*b[0]);
    thrust::reverse(gveca.begin(), gveca.end());
    thrust::transform(gveca.begin()+w, gveca.end(), gveca.begin(), gvecb.begin(), thrust::minus<Numeric>());
    thrust::reverse(gvecb.begin(), gvecb.end());

    // transfer data back to host
    thrust::copy(gvecb.begin(), gvecb.end(), pout);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// oneup - the gpu functor of constant value substraction, where negative values are set to 0
//
template <typename T>
struct oneup : public thrust::unary_function<T,T>
{
    T w;
	__host__ __device__
    oneup(T w) : w(w) {}

	__host__ __device__
    T operator()(const T& a) const
    {
    	if(a <= w)
    		return 0;
    	else
    		return a-w;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// gprup - host-gpu function for one wave and a horizontal line substraction 
// (negative values are set to 0)
// 
void gprup(PNumeric pint, PInteger a, PInteger b, PNumeric win1, PNumeric pout) {
	Numeric w1 = win1[0];

    // transfer data to the device
    thrust::device_vector<Numeric> gveca(a[0]*b[0]);
    thrust::copy(pint,pint+a[0]*b[0],gveca.begin());

    thrust::device_vector<Numeric> gvecb(a[0]*b[0]);
    thrust::transform(gveca.begin(), gveca.end(), gvecb.begin(), oneup<Numeric>(Numeric(w1)));

    // transfer data back to host
    thrust::copy(gvecb.begin(), gvecb.end(), pout);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// onedown - the gpu functor from constant value the wave substraction, 
// where negative values are set to 0
//
template <typename T>
struct onedown : public thrust::unary_function<T,T>
{
    T w;
	__host__ __device__
    onedown(T w) : w(w) {}

    __host__ __device__
    T operator()(const T& a) const
    {
    	if(a >= w)
    		return 0;
    	else
    		return w-a;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// gprdown - host-gpu function for a horizontal line and one wave substraction 
// (negative values are set to 0)
// 
void gprdown(PNumeric pint, PInteger a, PInteger b, PNumeric win1, PNumeric pout) {
	Numeric w1 = win1[0];

    // transfer data to the device
    thrust::device_vector<Numeric> gveca(a[0]*b[0]);
    thrust::copy(pint,pint+a[0]*b[0],gveca.begin());

    thrust::device_vector<Numeric> gvecb(a[0]*b[0]);
    thrust::transform(gveca.begin(), gveca.end(), gvecb.begin(), onedown<Numeric>(Numeric(w1)));

    // transfer data back to host
    thrust::copy(gvecb.begin(), gvecb.end(), pout);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// minus_by - the gpu functor of the wave and the constant value substraction square 
//
template<typename T>
struct  minus_by: public thrust::unary_function<T,T>
{
    T w;
	__host__ __device__
    minus_by(T w) : w(w) {}

   __host__ __device__
   T operator()(const T &x) const
   {
    return (x - w)*(x - w);
   }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// gprsdall - host-gpu function for a squared global average substraction sum 
// divided by a number of vectors   
// sqrt((x - globalavg)^2)/N
// 
void gprsdall(PNumeric pint, PInteger a, PInteger b, PNumeric pout) {

    // transfer data to the device
    thrust::device_vector<Numeric> gveca(a[0]*b[0]);
    thrust::copy(pint,pint+a[0]*b[0],gveca.begin());

    Numeric sumall = thrust::reduce(gveca.begin(), gveca.end())/(a[0]*b[0]);
    Numeric result = thrust::transform_reduce(gveca.begin(), gveca.end(),
                                            minus_by<Numeric>(Numeric(sumall)),
                                            0,
                                            thrust::plus<Numeric>());
    pout[0] = sqrt(result/(a[0]*b[0]));
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// gpravgall - host-gpu function for a global average of all vectors  
// 
void gpravgall(PNumeric pint, PInteger a, PInteger b, PNumeric pout) {

    // transfer data to the device
    thrust::device_vector<Numeric> gveca(a[0]*b[0]);
    thrust::copy(pint,pint+a[0]*b[0],gveca.begin());

	pout[0] = thrust::reduce(gveca.begin(), gveca.end())/(a[0]*b[0]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// kindcreate - the gpu functor of integer division by a constant m
//
struct kindcreate : public thrust::unary_function<Numeric,Numeric>
{
    Integer m, n;

    __host__ __device__
    kindcreate(Integer m, Integer n) : m(m), n(n) {}
    __host__ __device__
    Numeric operator()(const Numeric& a) const
    {
        //Integer ai=(int) ((int) a) % m;
        Integer av=(int) ((int) a) / m;
        //if(a > 0)
        return (int) av;
   }
};



////////////////////////////////////////////////////////////////////////////////////////////////////
// gprmovemax - host-gpu function for finding a global maximum horizontal line for each vector
//
void gprmovemax(PNumeric pint, PInteger a, PInteger b, PInteger win1, PNumeric pout) {

    // window size of the moving average
    //int w = win1[0];//difference window

    // transfer data to the device
    thrust::device_vector<Numeric> gveca(a[0]*b[0]);
    thrust::copy(pint,pint+a[0]*b[0],gveca.begin());

    thrust::device_vector<Integer> vindex(a[0]*b[0]);
    thrust::sequence(vindex.begin(),vindex.end(),0);
    thrust::device_vector<Numeric> gvecb(a[0]*b[0]);
    thrust::device_vector<Numeric> gvecc(a[0]*b[0]);
    thrust::transform(vindex.begin(), vindex.end(), gvecb.begin(), kindcreate(Integer(a[0]),Integer(b[0])));

    thrust::equal_to<Numeric> binary_pred;
    thrust::maximum<Numeric>   binary_op;
    thrust::inclusive_scan_by_key(gvecb.begin(), gvecb.end(), gveca.begin(), gvecc.begin(),binary_pred,binary_op);
    thrust::reverse(gvecc.begin(), gvecc.end());
    thrust::inclusive_scan_by_key(gvecb.begin(), gvecb.end(), gvecc.begin(), gveca.begin(),binary_pred,binary_op);
    thrust::reverse(gveca.begin(), gveca.end());

    // transfer data back to host
    thrust::copy(gveca.begin(), gveca.end(), pout);
    //thrust::copy(vindex.begin(), vindex.end(), pout);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// meanorig - gpu function for for the divided by m wave mean without baseline  
//
template <typename InputVector, typename OutputVector>
void meanorig(size_t m, size_t n, const InputVector& gveca, OutputVector& vout)
{
    thrust::device_vector<Numeric> orgbasoff(m*n);

    thrust::device_vector<Integer> vindex(m*n);
    thrust::sequence(vindex.begin(),vindex.end(),0);
    thrust::transform(vindex.begin(), vindex.end(), vindex.begin(), kindcreate(Integer(m),Integer(n)));
    thrust::device_vector<Numeric> gvecb(m*n);
//    thrust::device_vector<Numeric> gvecc(m*n);
    thrust::device_vector<Numeric> gvecd(m*n);

    double_moving_average(m,n,gveca, 150, gvecb);
    thrust::transform(gveca.begin(), gveca.end(), gvecb.begin(), orgbasoff.begin(), thrust::divides<Numeric>());
    thrust::transform(orgbasoff.begin(), orgbasoff.end(), orgbasoff.begin(), oneup<Numeric>(Numeric(1)));

    thrust::equal_to<Numeric> binary_pred;
    thrust::maximum<Numeric>  binary_max;
/*
    thrust::device_vector<Numeric> avgbasoff(a[0]*b[0]);
    double_moving_average(a[0],b[0],gveca, 150, gvecb);
    double_moving_average(a[0],b[0],gveca,  80, gvecc);
    thrust::transform(gvecc.begin(), gvecc.end(), gvecb.begin(), avgbasoff.begin(), thrust::divides<Numeric>());
    thrust::transform(avgbasoff.begin(), avgbasoff.end(), avgbasoff.begin(), oneup<Numeric>(Numeric(1)));


    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), avgbasoff.begin(), gvecb.begin(),binary_pred,binary_max);
    thrust::reverse(gvecb.begin(), gvecb.end());
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), gvecb.begin(), gvecc.begin(),binary_pred,binary_max);
    thrust::reverse(gvecc.begin(), gvecc.end());
*/
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), orgbasoff.begin(), gvecb.begin(),binary_pred,thrust::plus<Numeric>());
    thrust::reverse(gvecb.begin(), gvecb.end());
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), gvecb.begin(), gvecd.begin(),binary_pred,binary_max);
    thrust::reverse(gvecd.begin(), gvecd.end());
    thrust::fill(gvecb.begin(),gvecb.end(),m);
    thrust::transform(gvecd.begin(), gvecd.end(), gvecb.begin(), vout.begin(), thrust::divides<Numeric>());
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// gprmeanmax - host-gpu function for the divided by m wave mean without baseline  
//
void gprmeanmax(PNumeric pint, PInteger a, PInteger b, PInteger win1, PNumeric pout) {

    // window size of the moving average
    //int w = win1[0];//difference window

    // transfer data to the device
    thrust::device_vector<Numeric> gveca(a[0]*b[0]);
    thrust::copy(pint,pint+a[0]*b[0],gveca.begin());
    thrust::device_vector<Numeric> meanvec(a[0]*b[0]);

    meanorig(a[0], b[0], gveca, meanvec);
    // transfer data back to host
    thrust::copy(meanvec.begin(), meanvec.end(), pout);
    //thrust::copy(vindex.begin(), vindex.end(), pout);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// compare_zip - the gpu functor 
//
struct  compare_zip : public thrust::binary_function<Numeric2,Numeric2,Numeric>
{
     Numeric va,vb, da, db;

    __host__ __device__
    Numeric operator()(const Numeric2& a, const Numeric2& b) const
    {
		Numeric va=thrust::get<0>(a);
		Numeric da=thrust::get<1>(a);
		Numeric vb=thrust::get<0>(b);
		Numeric db=thrust::get<1>(b);
	    if(va > 0){
	    	if(da > 0 && db <0){
	    		return 1;
	    	}
	    	else
	    		return 0;
	    }
	    else
	    	return 0;
     }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// zipup - the gpu functor 
//
struct  zipup : public thrust::unary_function<Numeric2,Numeric>
{
     Numeric va,da;

    __host__ __device__
    Numeric operator()(const Numeric2& a) const
    {
		Numeric va=thrust::get<0>(a);
		Numeric da=thrust::get<1>(a);
    	if(va <= da)
    		return 0;
    	else
    		return va-da;
     }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// find_maxpeaks - gpu function for 
//
template <typename InputVector, typename OutputVector>
void find_maxpeaks(size_t m, size_t n, const InputVector& gveca, const Numeric& w1, const Numeric& w2, const Numeric& up, OutputVector& vout, OutputVector& avgbasoff, OutputVector& orgbasoff)
{
	//thrust::device_vector<Numeric> orgbasoff(m*n);

    thrust::device_vector<Integer> vindex(m*n);
    thrust::sequence(vindex.begin(),vindex.end(),0);
    thrust::transform(vindex.begin(), vindex.end(), vindex.begin(), kindcreate(Integer(m),Integer(n)));
    thrust::device_vector<Numeric> gvecb(m*n);
    thrust::device_vector<Numeric> gvecc(m*n);
    thrust::device_vector<Numeric> gvecd(m*n);

    double_moving_average(m,n,gveca, w2, gvecb);
    thrust::transform(gveca.begin(), gveca.end(), gvecb.begin(), orgbasoff.begin(), thrust::divides<Numeric>());

    if(up > 0)
    	thrust::transform(orgbasoff.begin(), orgbasoff.end(), orgbasoff.begin(), oneup<Numeric>(Numeric(1)));
    else
    	thrust::transform(orgbasoff.begin(), orgbasoff.end(), orgbasoff.begin(), onedown<Numeric>(Numeric(1)));

    thrust::equal_to<Numeric> binary_pred;
    thrust::maximum<Numeric>  binary_max;

    //thrust::device_vector<Numeric> avgbasoff(m*n);
    // up and down peaks from orig and avg divided by baseline
    double_moving_average(m,n,gveca, w2, gvecb);
    double_moving_average(m,n,gveca, w1, gvecc);
    thrust::transform(gvecc.begin(), gvecc.end(), gvecb.begin(), avgbasoff.begin(), thrust::divides<Numeric>());
    if(up > 0)
    	thrust::transform(avgbasoff.begin(), avgbasoff.end(), avgbasoff.begin(), oneup<Numeric>(Numeric(1)));
    else
    	thrust::transform(avgbasoff.begin(), avgbasoff.end(), avgbasoff.begin(), onedown<Numeric>(Numeric(1)));

    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), orgbasoff.begin(), gvecb.begin(),binary_pred,thrust::plus<Numeric>());
    thrust::reverse(gvecb.begin(), gvecb.end());
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), gvecb.begin(), gvecd.begin(),binary_pred,binary_max);
    thrust::reverse(gvecd.begin(), gvecd.end());
    thrust::fill(gvecb.begin(),gvecb.end(),m);
    thrust::transform(gvecd.begin(), gvecd.end(), gvecb.begin(), gvecd.begin(), thrust::divides<Numeric>());

    Numeric2Iterator first = thrust::make_zip_iterator(thrust::make_tuple(avgbasoff.begin(), gvecd.begin()));
    Numeric2Iterator last  = thrust::make_zip_iterator(thrust::make_tuple(avgbasoff.end(),   gvecd.end()));

    thrust::transform(first, last, gvecb.begin(), zipup());

	//substraction by 1
    thrust::transform(gvecb.begin()+1, gvecb.end(), gvecb.begin(), gvecc.begin(), thrust::minus<Numeric>());
    double_moving_average(m,n,gvecc, 20, gvecd);

    Numeric2Iterator first0 = thrust::make_zip_iterator(thrust::make_tuple(gvecb.begin(), gvecd.begin()));
    Numeric2Iterator first1 = thrust::make_zip_iterator(thrust::make_tuple(gvecb.begin() + 1, gvecd.begin() + 1));
    Numeric2Iterator last0  = thrust::make_zip_iterator(thrust::make_tuple(gvecb.end(),  gvecd.end()));

    thrust::transform(first0, last0, first1, vout.begin(), compare_zip());

}



////////////////////////////////////////////////////////////////////////////////////////////////////
// gprpeakmask - the host-gpu function for 
//
void gprpeakmask(PNumeric pint, PInteger a, PInteger b, PNumeric win1, PNumeric win2, PNumeric up, PNumeric pout) {
    size_t m = a[0];//row number
    size_t n = b[0];//column number
    int w1 = win1[0];//difference window
    int w2 = win2[0];//difference window

    // transfer data to the device
    thrust::device_vector<Numeric> gveca(a[0]*b[0]);
    thrust::copy(pint,pint+a[0]*b[0],gveca.begin());
    thrust::device_vector<Numeric> gvecb(a[0]*b[0]);
    thrust::device_vector<Numeric> gvecc(a[0]*b[0]);
    thrust::device_vector<Numeric> gvecd(a[0]*b[0]);

    find_maxpeaks(m,n,gveca,w1,w2,up[0],gvecb,gvecc,gvecd);
    // transfer data back to host
    thrust::copy(gvecb.begin(), gvecb.end(), pout);
    thrust::copy(gvecc.begin(), gvecc.end(), pout+m*n+1);
    thrust::copy(gvecd.begin(), gvecd.end(), pout+2*m*n+1);
}




////////////////////////////////////////////////////////////////////////////////////////////////////
// upmask - the gpu functor 
//
struct  upmask : public thrust::unary_function<Numeric,Numeric>
{
	Numeric out;
	__host__ __device__
    upmask(Numeric out) : out(out) {}
    __host__ __device__
    Numeric operator()(const Numeric& a)
    {
    	if(a > 0){
    		return out;
    	}
    	else
    	{
    		out=out+1;
    		return 0;
    	}
     }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// find_doublemaxpeaks - a gpu function for 
//
template <typename InputVector, typename OutputVector>
void find_doublemaxpeaks(size_t m, size_t n, const InputVector& idata, const Numeric& w1, const Numeric& w2, const Numeric& w3, OutputVector& pout, OutputVector& avgbasoffp, OutputVector& dout, OutputVector& avgbasoffd)
{
	thrust::device_vector<Numeric> orgbasoffp(m*n);
	thrust::device_vector<Numeric> orgbasoffd(m*n);

    thrust::device_vector<Numeric> data(m*n);
    thrust::device_vector<Numeric> data2(m*n);

    double_moving_average(m,n,idata, w2, data2);
    thrust::transform(idata.begin(), idata.end(), data2.begin(), data.begin(), thrust::divides<Numeric>());

    //original peaks up and down
    thrust::transform(data.begin(), data.end(), orgbasoffp.begin(), oneup<Numeric>(Numeric(1)));
    thrust::transform(data.begin(), data.end(), orgbasoffd.begin(), onedown<Numeric>(Numeric(1)));

    thrust::equal_to<Numeric> binary_pred;
    thrust::maximum<Numeric>  binary_max;
 
    double_moving_average(m,n,idata, w1, data);
    thrust::transform(data.begin(), data.end(), data2.begin(), data.begin(), thrust::divides<Numeric>());
	//average peaks up and down
    thrust::transform(data.begin(), data.end(), avgbasoffp.begin(), oneup<Numeric>(Numeric(1)));
    thrust::transform(data.begin(), data.end(), avgbasoffd.begin(), onedown<Numeric>(Numeric(1)));

	//the same number within each vector
    thrust::device_vector<Integer> vindex(m*n);
    thrust::sequence(vindex.begin(),vindex.end(),0);
    thrust::transform(vindex.begin(), vindex.end(), vindex.begin(), kindcreate(Integer(m),Integer(n)));

    //orgbasoffp, avgbasoffp
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), orgbasoffp.begin(), data.begin(), binary_pred,thrust::plus<Numeric>());
    thrust::reverse(data.begin(), data.end());
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), data.begin(), data2.begin(),binary_pred,binary_max);
    thrust::reverse(data2.begin(), data2.end());
    thrust::fill(data.begin(),data.end(),m/(NAVG/w1));
    thrust::transform(data2.begin(), data2.end(), data.begin(), data2.begin(), thrust::divides<Numeric>());

    Numeric2Iterator first = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffp.begin(), data2.begin()));
    Numeric2Iterator last  = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffp.end(),   data2.end()));

    thrust::transform(first, last, data.begin(), zipup());

    thrust::transform(data.begin()+1, data.end(), data.begin(), pout.begin(), thrust::minus<Numeric>());
    double_moving_average(m,n,pout, w3, data2);

    Numeric2Iterator first0 = thrust::make_zip_iterator(thrust::make_tuple(data.begin(), data2.begin()));
    Numeric2Iterator first1 = thrust::make_zip_iterator(thrust::make_tuple(data.begin() + 1, data2.begin() + 1));
    Numeric2Iterator last0  = thrust::make_zip_iterator(thrust::make_tuple(data.end(),  data2.end()));

    thrust::transform(first0, last0, first1, pout.begin(), compare_zip());

    //orgbasoffd, avgbasoffd
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), orgbasoffd.begin(), data.begin(), binary_pred,thrust::plus<Numeric>());
    thrust::reverse(data.begin(), data.end());
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), data.begin(), data2.begin(),binary_pred,binary_max);
    thrust::reverse(data2.begin(), data2.end());
    thrust::fill(data.begin(),data.end(),m/(NAVG/w1));
    thrust::transform(data2.begin(), data2.end(), data.begin(), data2.begin(), thrust::divides<Numeric>());

    first = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffd.begin(), data2.begin()));
    last  = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffd.end(),   data2.end()));

    thrust::transform(first, last, data.begin(), zipup());

    thrust::transform(data.begin()+1, data.end(), data.begin(), dout.begin(), thrust::minus<Numeric>());
    double_moving_average(m,n,dout, w3, data2);

    first0 = thrust::make_zip_iterator(thrust::make_tuple(data.begin(), data2.begin()));
    first1 = thrust::make_zip_iterator(thrust::make_tuple(data.begin() + 1, data2.begin() + 1));
    last0  = thrust::make_zip_iterator(thrust::make_tuple(data.end(),  data2.end()));
    
    thrust::transform(first0, last0, first1, dout.begin(), compare_zip());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// gprpeak2mask - the host-gpu function for 
//
void gprpeak2mask(PNumeric pint, PInteger a, PInteger b, PNumeric win1, PNumeric win2, PNumeric win3, PNumeric pout) {
    size_t m = a[0];//row number
    size_t n = b[0];//column number
    int w1 = win1[0];//window
    int w2 = win2[0];//baseline window
    int w3 = win3[0];//difference window

    // transfer data to the device
    thrust::device_vector<Numeric> gveca(pint,pint+a[0]*b[0]);
    thrust::device_vector<Numeric> gvecb(a[0]*b[0]);
    thrust::device_vector<Numeric> gvecc(a[0]*b[0]);
    thrust::device_vector<Numeric> gvecd(a[0]*b[0]);
    thrust::device_vector<Numeric> gvece(a[0]*b[0]);

    find_doublemaxpeaks(m,n,gveca,w1,w2,w3,gvecb,gvecc,gvecd,gvece);
    // transfer data back to host
    thrust::copy(gvecb.begin(), gvecb.end(), pout);
    thrust::copy(gvecc.begin(), gvecc.end(), pout+m*n+1);
    thrust::copy(gvecd.begin(), gvecd.end(), pout+2*m*n+1);
    thrust::copy(gvece.begin(), gvece.end(), pout+3*m*n+1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// avg find peaks 
// 
template <typename InputVector, typename OutputVector>
void avg_find_peaks(size_t m, size_t n, const Numeric& w1, const Numeric& w2, const Numeric& w3, 
InputVector& orgbasoff, InputVector& avgbasoff,OutputVector& out)
{
  typedef typename InputVector::value_type T;
  thrust::equal_to<Numeric> binary_pred;
  thrust::maximum<Numeric>  binary_max;

  thrust::device_vector<Numeric> data(m*n);
  thrust::device_vector<Numeric> data2(m*n);
  
  //the same number within each vector
  thrust::device_vector<Integer> vindex(m*n);
  thrust::sequence(vindex.begin(),vindex.end(),0);
  thrust::transform(vindex.begin(), vindex.end(), vindex.begin(), kindcreate(Integer(m),Integer(n)));
  
  thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), orgbasoff.begin(), data.begin(), binary_pred,thrust::plus<Numeric>());
  thrust::reverse(data.begin(), data.end());
  thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), data.begin(), data2.begin(),binary_pred,binary_max);
  thrust::reverse(data2.begin(), data2.end());
  thrust::fill(data.begin(),data.end(),m/(NAVG/w1));
  thrust::transform(data2.begin(), data2.end(), data.begin(), data2.begin(), thrust::divides<Numeric>());
  
  	//cout << "peaks divided by average:" << endl;
  	//printvec(1,NDATA,data2);
  	 
  Numeric2Iterator first = thrust::make_zip_iterator(thrust::make_tuple(avgbasoff.begin(), data2.begin()));
  Numeric2Iterator last  = thrust::make_zip_iterator(thrust::make_tuple(avgbasoff.end(),   data2.end()));

  thrust::transform(first, last, data.begin(), zipup());

  	//cout << "peaks zipup:" << endl;
  	//printvec(1,NDATA,data);

  thrust::transform(data.begin()+1, data.end(), data.begin(), out.begin(), thrust::minus<Numeric>());
  double_moving_average(m,n,out, w3, data2);

  Numeric2Iterator first0 = thrust::make_zip_iterator(thrust::make_tuple(data.begin(), data2.begin()));
  Numeric2Iterator first1 = thrust::make_zip_iterator(thrust::make_tuple(data.begin() + 1, data2.begin() + 1));
  Numeric2Iterator last0  = thrust::make_zip_iterator(thrust::make_tuple(data.end(),  data2.end()));

  thrust::transform(first0, last0, first1, out.begin(), compare_zip());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// segregate peaks up and down
// 
template <typename InputVector, typename OutputVector>
void create_peaks(InputVector& data, InputVector& base, OutputVector& wavbasoffp, OutputVector& wavbasoffd)
{
	typedef typename InputVector::value_type T;
	thrust::device_vector<Numeric> data2(data.begin(),data.end());
   	thrust::transform(data.begin(), data.end(), base.begin(), data2.begin(), thrust::divides<Numeric>());
  	thrust::transform(data2.begin(), data2.end(), wavbasoffp.begin(), oneup<Numeric>(Numeric(1)));
  	thrust::transform(data2.begin(), data2.end(), wavbasoffd.begin(), onedown<Numeric>(Numeric(1)));
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// mask01 - the gpu functor of making mask, where values greater than w are set to 1
//
template <typename T>
struct mask01 : public thrust::unary_function<T,T>
{
    T w;
	__host__ __device__
    mask01(T w) : w(w) {}

	__host__ __device__
    T operator()(const T& a) const
    {
    	if(a > w)
    		return 1;
    	else
    		return 0;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// tone - the gpu functor of making 1, where values are equal to 1
//
template <typename T>
struct tone : public thrust::unary_function<T,T>
{
    T w;
	__host__ __device__
    tone(T w) : w(w) {}

	__host__ __device__
    T operator()(const T& a) const
    {
    	if(a > w)
    		return a;
    	else
    		return 1;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// segmented by peaks sums 
// 
template <typename InputVector, typename OutputVector>
void segmented_peak_sums(size_t m, size_t n, const InputVector& idata, OutputVector& data)
{
  typedef typename InputVector::value_type T;

  thrust::device_vector<Numeric> peakmask(m*n);

  thrust::transform(idata.begin(), idata.end(), peakmask.begin(), mask01<Numeric>(Integer(0)));
  //thrust::device_vector<Numeric> peakbase(peakmask.begin(), peakmask.end());
  //thrust::inclusive_scan_by_key(peakmask.begin(), peakmask.end(), peakbase.begin(),peakbase.begin());
  thrust::inclusive_scan_by_key(peakmask.begin(), peakmask.end(), idata.begin(),data.begin());
  //thrust::reverse(peakbase.begin(), peakbase.end());
  thrust::reverse(data.begin(), data.end());
  thrust::reverse(peakmask.begin(), peakmask.end());
  thrust::equal_to<Numeric> binary_pred;
  thrust::maximum<Numeric>  binary_max;
  thrust::inclusive_scan_by_key(peakmask.begin(), peakmask.end(), data.begin(), data.begin(),binary_pred,binary_max);
  //thrust::inclusive_scan_by_key(peakmask.begin(), peakmask.end(), peakbase.begin(), peakbase.begin(),binary_pred,binary_max);
  //thrust::reverse(peakbase.begin(), peakbase.end());
  thrust::reverse(data.begin(), data.end());
  //thrust::transform(peakbase.begin(), peakbase.end(), peakbase.begin(), tone<Numeric>(Integer(0)));
  //cout << "data of peakbase:" << endl;
  //printvec(1,NDATA,peakbase); 
  //thrust::transform(data.begin(), data.end(), peakbase.begin(), data.begin(), thrust::divides<Numeric>());

}

////////////////////////////////////////////////////////////////////////////////////////////////////
// find_doublemaxpeaks_opt - a gpu function for 
//
template <typename InputVector, typename OutputVector>
void find_doublemaxpeaks_opt(size_t m, size_t n, const InputVector& idata, const Numeric& w1, const Numeric& w2, const Numeric& w3, 
OutputVector& pout, OutputVector& avgbasoffp, OutputVector& pintegralp, OutputVector& dout, OutputVector& avgbasoffd, OutputVector& pintegrald)
{
	thrust::device_vector<Numeric> orgbasoffp(m*n);
	thrust::device_vector<Numeric> orgbasoffd(m*n);

    thrust::device_vector<Numeric> data(m*n);
    thrust::device_vector<Numeric> data2(m*n);

    double_moving_average(m,n,idata, w2, data2);
    thrust::transform(idata.begin(), idata.end(), data2.begin(), data.begin(), thrust::divides<Numeric>());

    //original peaks up and down
    thrust::transform(data.begin(), data.end(), orgbasoffp.begin(), oneup<Numeric>(Numeric(1)));
    thrust::transform(data.begin(), data.end(), orgbasoffd.begin(), onedown<Numeric>(Numeric(1)));

    thrust::equal_to<Numeric> binary_pred;
    thrust::maximum<Numeric>  binary_max;
 
    double_moving_average(m,n,idata, w1, data);
    thrust::transform(data.begin(), data.end(), data2.begin(), data.begin(), thrust::divides<Numeric>());
	//average peaks up and down
    thrust::transform(data.begin(), data.end(), avgbasoffp.begin(), oneup<Numeric>(Numeric(1)));
    thrust::transform(data.begin(), data.end(), avgbasoffd.begin(), onedown<Numeric>(Numeric(1)));

	//the same number within each vector
    thrust::device_vector<Integer> vindex(m*n);
    thrust::sequence(vindex.begin(),vindex.end(),0);
    thrust::transform(vindex.begin(), vindex.end(), vindex.begin(), kindcreate(Integer(m),Integer(n)));

    //orgbasoffp, avgbasoffp
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), orgbasoffp.begin(), data.begin(), binary_pred,thrust::plus<Numeric>());
    thrust::reverse(data.begin(), data.end());
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), data.begin(), data2.begin(),binary_pred,binary_max);
    thrust::reverse(data2.begin(), data2.end());
    thrust::fill(data.begin(),data.end(),m/(NAVG/w1));
    thrust::transform(data2.begin(), data2.end(), data.begin(), data2.begin(), thrust::divides<Numeric>());

    Numeric2Iterator first = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffp.begin(), data2.begin()));
    Numeric2Iterator last  = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffp.end(),   data2.end()));

    thrust::transform(first, last, data.begin(), zipup());

    thrust::transform(data.begin()+1, data.end(), data.begin(), pout.begin(), thrust::minus<Numeric>());
    double_moving_average(m,n,pout, w3, data2);

    Numeric2Iterator first0 = thrust::make_zip_iterator(thrust::make_tuple(data.begin(), data2.begin()));
    Numeric2Iterator first1 = thrust::make_zip_iterator(thrust::make_tuple(data.begin() + 1, data2.begin() + 1));
    Numeric2Iterator last0  = thrust::make_zip_iterator(thrust::make_tuple(data.end(),  data2.end()));

    thrust::transform(first0, last0, first1, pout.begin(), compare_zip());

    //orgbasoffd, avgbasoffd
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), orgbasoffd.begin(), data.begin(), binary_pred,thrust::plus<Numeric>());
    thrust::reverse(data.begin(), data.end());
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), data.begin(), data2.begin(),binary_pred,binary_max);
    thrust::reverse(data2.begin(), data2.end());
    thrust::fill(data.begin(),data.end(),m/(NAVG/w1));
    thrust::transform(data2.begin(), data2.end(), data.begin(), data2.begin(), thrust::divides<Numeric>());

    first = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffd.begin(), data2.begin()));
    last  = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffd.end(),   data2.end()));

    thrust::transform(first, last, data.begin(), zipup());

    thrust::transform(data.begin()+1, data.end(), data.begin(), dout.begin(), thrust::minus<Numeric>());
    double_moving_average(m,n,dout, w3, data2);

    first0 = thrust::make_zip_iterator(thrust::make_tuple(data.begin(), data2.begin()));
    first1 = thrust::make_zip_iterator(thrust::make_tuple(data.begin() + 1, data2.begin() + 1));
    last0  = thrust::make_zip_iterator(thrust::make_tuple(data.end(),  data2.end()));
    
    thrust::transform(first0, last0, first1, dout.begin(), compare_zip());
    segmented_peak_sums(m, n, avgbasoffp, pintegralp);
    segmented_peak_sums(m, n, avgbasoffd, pintegrald);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// gprpeak2maskopt - the host-gpu function for 
//
void gprpeak2maskopt(PNumeric pint, PInteger a, PInteger b, PNumeric win1, PNumeric win2, PNumeric win3, PNumeric pout) {
    size_t m = a[0];//row number
    size_t n = b[0];//column number
    int w1 = win1[0];//window
    int w2 = win2[0];//baseline window
    int w3 = win3[0];//difference window

    // transfer data to the device
    thrust::device_vector<Numeric> gveca(pint,pint+a[0]*b[0]);
    thrust::device_vector<Numeric> gvecb(a[0]*b[0]);
    thrust::device_vector<Numeric> gvecc(a[0]*b[0]);
    thrust::device_vector<Numeric> gvecd(a[0]*b[0]);
    thrust::device_vector<Numeric> gvece(a[0]*b[0]);
    thrust::device_vector<Numeric> gvecf(a[0]*b[0]);
    thrust::device_vector<Numeric> gvecg(a[0]*b[0]);

    find_doublemaxpeaks_opt(m,n,gveca,w1,w2,w3,gvecb,gvecc,gvecd,gvece,gvecf,gvecg);
    // transfer data back to host
    thrust::copy(gvecb.begin(), gvecb.end(), pout);
    thrust::copy(gvecc.begin(), gvecc.end(), pout+m*n+1);
    thrust::copy(gvecd.begin(), gvecd.end(), pout+2*m*n+1);
    thrust::copy(gvece.begin(), gvece.end(), pout+3*m*n+1);
    thrust::copy(gvecf.begin(), gvecf.end(), pout+4*m*n+1);
    thrust::copy(gvecg.begin(), gvecg.end(), pout+5*m*n+1);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// segmented by peaks multiplied by lambdadiff sums 
// 
template <typename InputVector, typename OutputVector>
void lambda_peak_sums(size_t m, size_t n, const InputVector& ldiff, const InputVector& idata, OutputVector& data)
{
  typedef typename InputVector::value_type T;

  thrust::device_vector<Numeric> peakmask(m*n);
  thrust::device_vector<Numeric> sdata(m*n);

  thrust::transform(idata.begin(), idata.end(), peakmask.begin(), mask01<Numeric>(Integer(0)));
  thrust::transform(ldiff.begin(), ldiff.end(), idata.begin(), sdata.begin(), thrust::multiplies<Numeric>());  
  thrust::inclusive_scan_by_key(peakmask.begin(), peakmask.end(), sdata.begin(),data.begin());
  thrust::reverse(data.begin(), data.end());
  thrust::reverse(peakmask.begin(), peakmask.end());
  thrust::equal_to<Numeric> binary_pred;
  thrust::maximum<Numeric>  binary_max;
  thrust::inclusive_scan_by_key(peakmask.begin(), peakmask.end(), data.begin(), data.begin(),binary_pred,binary_max);
  thrust::reverse(data.begin(), data.end());
  
  //thrust::transform(peakbase.begin(), peakbase.end(), peakbase.begin(), tone<Numeric>(Integer(0)));
  //cout << "data of peakbase:" << endl;
  //printvec(1,NDATA,peakbase); 
  //thrust::transform(data.begin(), data.end(), peakbase.begin(), data.begin(), thrust::divides<Numeric>());

}

////////////////////////////////////////////////////////////////////////////////////////////////////
// find_doublemaxpeaks_lambda - a gpu function for 
//
template <typename InputVector, typename OutputVector>
void find_doublemaxpeaks_lambda(size_t m, size_t n, const InputVector& lambda, const InputVector& idata, const Numeric& w1, const Numeric& w2, const Numeric& w3, 
OutputVector& pout, OutputVector& avgbasoffp, OutputVector& pintegralp, OutputVector& dout, OutputVector& avgbasoffd, OutputVector& pintegrald)
{
	thrust::device_vector<Numeric> ldiff(m*n);
	thrust::transform(lambda.begin()+1, lambda.end(), lambda.begin(), ldiff.begin(), thrust::minus<Numeric>());

	thrust::device_vector<Numeric> orgbasoffp(m*n);
	thrust::device_vector<Numeric> orgbasoffd(m*n);

    thrust::device_vector<Numeric> data(m*n);
    thrust::device_vector<Numeric> data2(m*n);

    double_moving_average(m,n,idata, w2, data2);
    thrust::transform(idata.begin(), idata.end(), data2.begin(), data.begin(), thrust::divides<Numeric>());

    //original peaks up and down
    thrust::transform(data.begin(), data.end(), orgbasoffp.begin(), oneup<Numeric>(Numeric(1)));
    thrust::transform(data.begin(), data.end(), orgbasoffd.begin(), onedown<Numeric>(Numeric(1)));

    thrust::equal_to<Numeric> binary_pred;
    thrust::maximum<Numeric>  binary_max;
 
    double_moving_average(m,n,idata, w1, data);
    thrust::transform(data.begin(), data.end(), data2.begin(), data.begin(), thrust::divides<Numeric>());
	//average peaks up and down
    thrust::transform(data.begin(), data.end(), avgbasoffp.begin(), oneup<Numeric>(Numeric(1)));
    thrust::transform(data.begin(), data.end(), avgbasoffd.begin(), onedown<Numeric>(Numeric(1)));

	//the same number within each vector
    thrust::device_vector<Integer> vindex(m*n);
    thrust::sequence(vindex.begin(),vindex.end(),0);
    thrust::transform(vindex.begin(), vindex.end(), vindex.begin(), kindcreate(Integer(m),Integer(n)));

    //orgbasoffp, avgbasoffp
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), orgbasoffp.begin(), data.begin(), binary_pred,thrust::plus<Numeric>());
    thrust::reverse(data.begin(), data.end());
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), data.begin(), data2.begin(),binary_pred,binary_max);
    thrust::reverse(data2.begin(), data2.end());
    thrust::fill(data.begin(),data.end(),m/(NAVG/w1));
    thrust::transform(data2.begin(), data2.end(), data.begin(), data2.begin(), thrust::divides<Numeric>());

    Numeric2Iterator first = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffp.begin(), data2.begin()));
    Numeric2Iterator last  = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffp.end(),   data2.end()));

    thrust::transform(first, last, data.begin(), zipup());

    thrust::transform(data.begin()+1, data.end(), data.begin(), pout.begin(), thrust::minus<Numeric>());
    double_moving_average(m,n,pout, w3, data2);

    Numeric2Iterator first0 = thrust::make_zip_iterator(thrust::make_tuple(data.begin(), data2.begin()));
    Numeric2Iterator first1 = thrust::make_zip_iterator(thrust::make_tuple(data.begin() + 1, data2.begin() + 1));
    Numeric2Iterator last0  = thrust::make_zip_iterator(thrust::make_tuple(data.end(),  data2.end()));

    thrust::transform(first0, last0, first1, pout.begin(), compare_zip());

    //orgbasoffd, avgbasoffd
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), orgbasoffd.begin(), data.begin(), binary_pred,thrust::plus<Numeric>());
    thrust::reverse(data.begin(), data.end());
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), data.begin(), data2.begin(),binary_pred,binary_max);
    thrust::reverse(data2.begin(), data2.end());
    thrust::fill(data.begin(),data.end(),m/(NAVG/w1));
    thrust::transform(data2.begin(), data2.end(), data.begin(), data2.begin(), thrust::divides<Numeric>());

    first = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffd.begin(), data2.begin()));
    last  = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffd.end(),   data2.end()));

    thrust::transform(first, last, data.begin(), zipup());

    thrust::transform(data.begin()+1, data.end(), data.begin(), dout.begin(), thrust::minus<Numeric>());
    double_moving_average(m,n,dout, w3, data2);

    first0 = thrust::make_zip_iterator(thrust::make_tuple(data.begin(), data2.begin()));
    first1 = thrust::make_zip_iterator(thrust::make_tuple(data.begin() + 1, data2.begin() + 1));
    last0  = thrust::make_zip_iterator(thrust::make_tuple(data.end(),  data2.end()));
    
    thrust::transform(first0, last0, first1, dout.begin(), compare_zip());
    lambda_peak_sums(m, n, ldiff, avgbasoffp, pintegralp);
    lambda_peak_sums(m, n, ldiff, avgbasoffd, pintegrald);
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// gprpeaklambda2mask - the host-gpu function for 
//
void gprpeaklambda2mask(PNumeric lint, PNumeric pint, PInteger a, PInteger b, PNumeric win1, PNumeric win2, PNumeric win3, PNumeric pout) {
    size_t m = a[0];//row number
    size_t n = b[0];//column number
    int w1 = win1[0];//window
    int w2 = win2[0];//baseline window
    int w3 = win3[0];//difference window

    // transfer data to the device
    thrust::device_vector<Numeric> lambda(lint,lint+a[0]*b[0]);
    thrust::device_vector<Numeric> gveca(pint,pint+a[0]*b[0]);
    thrust::device_vector<Numeric> gvecb(a[0]*b[0]);
    thrust::device_vector<Numeric> gvecc(a[0]*b[0]);
    thrust::device_vector<Numeric> gvecd(a[0]*b[0]);
    thrust::device_vector<Numeric> gvece(a[0]*b[0]);
    thrust::device_vector<Numeric> gvecf(a[0]*b[0]);
    thrust::device_vector<Numeric> gvecg(a[0]*b[0]);

    find_doublemaxpeaks_lambda(m,n,lambda,gveca,w1,w2,w3,gvecb,gvecc,gvecd,gvece,gvecf,gvecg);
    // transfer data back to host
    thrust::copy(gvecb.begin(), gvecb.end(), pout);
    thrust::copy(gvecc.begin(), gvecc.end(), pout+m*n+1);
    thrust::copy(gvecd.begin(), gvecd.end(), pout+2*m*n+1);
    thrust::copy(gvece.begin(), gvece.end(), pout+3*m*n+1);
    thrust::copy(gvecf.begin(), gvecf.end(), pout+4*m*n+1);
    thrust::copy(gvecg.begin(), gvecg.end(), pout+5*m*n+1);
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// half_peak_width
// 
template <typename InputVector, typename OutputVector>
void half_peak_width(size_t m, size_t n, const InputVector& ldiff, const InputVector& idata, OutputVector& data)
{
  typedef typename InputVector::value_type T;

  thrust::device_vector<Numeric> peakmask(m*n);
  thrust::transform(idata.begin(), idata.end(), peakmask.begin(), mask01<Numeric>(Integer(0)));
  //max for peakmask
  thrust::equal_to<Numeric> binary_pred;
  thrust::maximum<Numeric>  binary_max;
  thrust::inclusive_scan_by_key(peakmask.begin(), peakmask.end(), idata.begin(), data.begin(),binary_pred,binary_max);
  thrust::reverse(data.begin(), data.end());
  thrust::reverse(peakmask.begin(), peakmask.end());
  thrust::inclusive_scan_by_key(peakmask.begin(), peakmask.end(), data.begin(), data.begin(),binary_pred,binary_max);
  thrust::reverse(data.begin(), data.end());
  thrust::device_vector<Numeric> halfdata(m*n);
  thrust::fill(halfdata.begin(),halfdata.end(),2);
  thrust::transform(data.begin(), data.end(), halfdata.begin(), data.begin(), thrust::divides<Numeric>());
  //minus half max
  thrust::transform(idata.begin(), idata.end(), data.begin(), halfdata.begin(), thrust::minus<Numeric>());
  thrust::transform(halfdata.begin(), halfdata.end(), peakmask.begin(), mask01<Numeric>(Integer(0)));
  thrust::plus<Numeric>  binary_plus;
  //width in mask
  thrust::inclusive_scan_by_key(peakmask.begin(), peakmask.end(), ldiff.begin(), data.begin(),binary_pred,binary_plus);
  thrust::reverse(data.begin(), data.end());
  thrust::reverse(peakmask.begin(), peakmask.end());
  thrust::inclusive_scan_by_key(peakmask.begin(), peakmask.end(), data.begin(), data.begin(),binary_pred,binary_max);
  thrust::reverse(data.begin(), data.end());
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// find_doublemaxpeaks_lambdahalf - a gpu function for 
//
template <typename InputVector, typename OutputVector>
void find_doublemaxpeaks_lambdahalf(size_t m, size_t n, const InputVector& lambda, const InputVector& idata, const Numeric& w1, const Numeric& w2, const Numeric& w3, 
OutputVector& pout, OutputVector& avgbasoffp, OutputVector& pintegralp, OutputVector& halfwp, OutputVector& dout, OutputVector& avgbasoffd, OutputVector& pintegrald, OutputVector& halfwd, const Numeric& navg)
{
	thrust::device_vector<Numeric> ldiff(m*n);
	thrust::transform(lambda.begin()+1, lambda.end(), lambda.begin(), ldiff.begin(), thrust::minus<Numeric>());

	thrust::device_vector<Numeric> orgbasoffp(m*n);
	thrust::device_vector<Numeric> orgbasoffd(m*n);

    thrust::device_vector<Numeric> data(m*n);
    thrust::device_vector<Numeric> data2(m*n);

    double_moving_average(m,n,idata, w2, data2);
    thrust::transform(idata.begin(), idata.end(), data2.begin(), data.begin(), thrust::divides<Numeric>());

    //original peaks up and down
    thrust::transform(data.begin(), data.end(), orgbasoffp.begin(), oneup<Numeric>(Numeric(1)));
    thrust::transform(data.begin(), data.end(), orgbasoffd.begin(), onedown<Numeric>(Numeric(1)));

    thrust::equal_to<Numeric> binary_pred;
    thrust::maximum<Numeric>  binary_max;
 
    double_moving_average(m,n,idata, w1, data);
    thrust::transform(data.begin(), data.end(), data2.begin(), data.begin(), thrust::divides<Numeric>());
	//average peaks up and down
    thrust::transform(data.begin(), data.end(), avgbasoffp.begin(), oneup<Numeric>(Numeric(1)));
    thrust::transform(data.begin(), data.end(), avgbasoffd.begin(), onedown<Numeric>(Numeric(1)));

	//the same number within each vector
    thrust::device_vector<Integer> vindex(m*n);
    thrust::sequence(vindex.begin(),vindex.end(),0);
    thrust::transform(vindex.begin(), vindex.end(), vindex.begin(), kindcreate(Integer(m),Integer(n)));

    //orgbasoffp, avgbasoffp
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), orgbasoffp.begin(), data.begin(), binary_pred,thrust::plus<Numeric>());
    thrust::reverse(data.begin(), data.end());
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), data.begin(), data2.begin(),binary_pred,binary_max);
    thrust::reverse(data2.begin(), data2.end());
    thrust::fill(data.begin(),data.end(),m/(navg/w1));
    thrust::transform(data2.begin(), data2.end(), data.begin(), data2.begin(), thrust::divides<Numeric>());

    Numeric2Iterator first = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffp.begin(), data2.begin()));
    Numeric2Iterator last  = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffp.end(),   data2.end()));

    thrust::transform(first, last, pout.begin(), zipup());

	//double_moving_average(m,n,data, w3, pout);

    thrust::transform(pout.begin()+1, pout.end(), pout.begin(), data.begin(), thrust::minus<Numeric>());
thrust::copy(data.begin(), data.end(), avgbasoffp.begin());    
    Numeric2Iterator first0 = thrust::make_zip_iterator(thrust::make_tuple(data.begin(), data2.begin()));
    Numeric2Iterator first1 = thrust::make_zip_iterator(thrust::make_tuple(data.begin() + 1, data2.begin() + 1));
    Numeric2Iterator last0  = thrust::make_zip_iterator(thrust::make_tuple(data.end(),  data2.end()));

    thrust::transform(first0, last0, first1, pout.begin(), compare_zip());

    //orgbasoffd, avgbasoffd
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), orgbasoffd.begin(), data.begin(), binary_pred,thrust::plus<Numeric>());
    thrust::reverse(data.begin(), data.end());
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), data.begin(), data2.begin(),binary_pred,binary_max);
    thrust::reverse(data2.begin(), data2.end());
    thrust::fill(data.begin(),data.end(),m/(navg/w1));
    thrust::transform(data2.begin(), data2.end(), data.begin(), data2.begin(), thrust::divides<Numeric>());

    first = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffd.begin(), data2.begin()));
    last  = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffd.end(),   data2.end()));

    thrust::transform(first, last, dout.begin(), zipup());

    //double_moving_average(m,n,data, w3, dout);

    thrust::transform(dout.begin()+1, dout.end(), dout.begin(), data.begin(), thrust::minus<Numeric>());
thrust::copy(data.begin(), data.end(), avgbasoffd.begin());
    first0 = thrust::make_zip_iterator(thrust::make_tuple(data.begin(), data2.begin()));
    first1 = thrust::make_zip_iterator(thrust::make_tuple(data.begin() + 1, data2.begin() + 1));
    last0  = thrust::make_zip_iterator(thrust::make_tuple(data.end(),  data2.end()));
    
    thrust::transform(first0, last0, first1, dout.begin(), compare_zip());
    //lambda_peak_sums(m, n, ldiff, avgbasoffp, pintegralp);
    //lambda_peak_sums(m, n, ldiff, avgbasoffd, pintegrald);
    //half_peak_width(m, n, ldiff, avgbasoffp, halfwp);
    //half_peak_width(m, n, ldiff, avgbasoffd, halfwd);
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// gprpeaklambda2halfmask - the host-gpu function for 
//
void gprpeaklambda2halfmask(PNumeric lint, PNumeric pint, PInteger a, PInteger b, PNumeric win1, PNumeric win2, PNumeric win3, PNumeric pout, PInteger navg) {
    size_t m = a[0];//row number
    size_t n = b[0];//column number
    int w1 = win1[0];//window
    int w2 = win2[0];//baseline window
    int w3 = win3[0];//difference window
	int navg0 = navg[0];

    // transfer data to the device
    thrust::device_vector<Numeric> lambda(lint,lint+a[0]*b[0]);
    thrust::device_vector<Numeric> gveca(pint,pint+a[0]*b[0]);
    thrust::device_vector<Numeric> gvecb(a[0]*b[0]);
    thrust::device_vector<Numeric> gvecc(a[0]*b[0]);
    thrust::device_vector<Numeric> gvecd(a[0]*b[0]);
    thrust::device_vector<Numeric> gvece(a[0]*b[0]);
    thrust::device_vector<Numeric> gvecf(a[0]*b[0]);
    thrust::device_vector<Numeric> gvecg(a[0]*b[0]);
    thrust::device_vector<Numeric> gvech(a[0]*b[0]);
    thrust::device_vector<Numeric> gveci(a[0]*b[0]);
    
    find_doublemaxpeaks_lambdahalf(m,n,lambda,gveca,w1,w2,w3,gvecb,gvecc,gvecd,gvece,gvecf,gvecg,gvech,gveci,navg0);
    // transfer data back to host
    thrust::copy(gvecb.begin(), gvecb.end(), pout);
    thrust::copy(gvecc.begin(), gvecc.end(), pout+m*n+1);
    thrust::copy(gvecd.begin(), gvecd.end(), pout+2*m*n+1);
    thrust::copy(gvece.begin(), gvece.end(), pout+3*m*n+1);
    thrust::copy(gvecf.begin(), gvecf.end(), pout+4*m*n+1);
    thrust::copy(gvecg.begin(), gvecg.end(), pout+5*m*n+1);
    thrust::copy(gvech.begin(), gvech.end(), pout+6*m*n+1);
    thrust::copy(gveci.begin(), gveci.end(), pout+7*m*n+1);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
// exemplary gpu functors
//
template <typename T>
struct is_less_than_zero
{
   __host__ __device__
   bool operator()(T x)
   {
      return x < 0;
   }
};

template <typename T>
struct is_greater_than
{
   T w;
   __host__ __device__
   is_greater_than(T w) : w(w) {}
   __host__ __device__
   bool operator()(T x)
   {
      return x > w;
   }
};


template <typename T>
struct minus_and_divide : public thrust::binary_function<T,T,T>
{
    T w;

    minus_and_divide(T w) : w(w) {}

    __host__ __device__
    T operator()(const T& a, const T& b) const
    {
        return (a - b) / w;
    }
};

template <typename T>
struct minus_and_divide_w : public thrust::binary_function<T,T,T>
{
    T w, maxb;

    __host__ __device__
    minus_and_divide_w(T w, T maxb) : w(w), maxb(maxb) {}

    __host__ __device__
    T operator()(const T& a, const T& b) const
    {
    	if(a < b)
        return (a - b) / w;
	else 
	return (a - maxb) / (w - a % maxb);
    }
};

// convert a linear index to a linear index in the transpose 
struct transpose_index : public thrust::unary_function<size_t,size_t>
{
    size_t m, n;

    __host__ __device__
    transpose_index(size_t _m, size_t _n) : m(_m), n(_n) {}

    __host__ __device__
    size_t operator()(size_t linear_index)
    {
        size_t j = linear_index / m;
        size_t i = linear_index % m;

        return j + i * n;
    }
};

// convert a linear index to a row index
struct column_index : public thrust::unary_function<size_t,size_t>
{
    size_t n;
    
    __host__ __device__
    column_index(size_t _n) : n(_n) {}

    __host__ __device__
    size_t operator()(size_t i)
    {
        return i / n;
    }
};

// convert a linear index to a row index
struct binary_index : public thrust::unary_function<size_t,size_t>
{
    size_t n;
    
    __host__ __device__
    binary_index(size_t _n) : n(_n) {}

    __host__ __device__
    size_t operator()(size_t i)
    {
        return (i / n) % 2;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// exemplary gpu functions
//

// transpose an M-by-N array
template <typename T>
void transpose(size_t m, size_t n, thrust::device_vector<T>& src, thrust::device_vector<T>& dst)
{
    thrust::counting_iterator<size_t> indices(0);
    
    thrust::gather(thrust::make_transform_iterator(indices, transpose_index(n, m)),
                   thrust::make_transform_iterator(indices, transpose_index(n, m)) + dst.size(),
                   src.begin(),
                   dst.begin());
}


// scan the rows of an M-by-N array
template <typename T>
void scan_horizontally(size_t m, size_t n, thrust::device_vector<T>& d_data)
{
    thrust::counting_iterator<size_t> indices(0);

    thrust::inclusive_scan_by_key(d_data.begin(), d_data.end(),
                                  thrust::make_transform_iterator(indices, column_index(m)),
                                  d_data.begin());
}




template <typename T>
void sumvec(thrust::device_vector<T>& gvec, Numeric& out)
{
    thrust::reduce(gvec.begin(), gvec.end(), out);
}

