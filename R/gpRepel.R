#####################################################################
##  gpRepel : An R package for GPU computing
##
##  This program is free software; you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation; version 3 of the License.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program; if not, write to the Free Software
##  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA


#####################################################################
# InitPackage
#
InitPackage <- function() {
    return ("gpRepel")
}


#####################################################################
# gpRPostmave - the simple average: window w points forward, 
# after a given point
#
gpRPostmave <- function(points,w) {

    pkg <- InitPackage()

    points <- as.matrix(points)
    num <- nrow(points)
    dim <- ncol(points)
     
    d <- .C("gprpostmave",
            as.single(points),
            as.integer(num),
            as.integer(dim),
	    as.integer(w),
            d = single(num*dim),
            NAOK = TRUE,
            PACKAGE = pkg)$d

    return (matrix(d,num,dim))
}



#####################################################################
# gpRPremave
#
gpRPremave <- function(points,w) {

    pkg <- InitPackage()

    points <- as.matrix(points)
    num <- nrow(points)
    dim <- ncol(points)
     
    d <- .C("gprpremave",
            as.single(points),
            as.integer(num),
            as.integer(dim),
	    as.integer(w),
            d = single(num*dim),
            NAOK = TRUE,
            PACKAGE = pkg)$d

    return (matrix(d,num,dim))
}

#####################################################################
# gpRMoverage
#
gpRMoverage <- function(points,w) {

    pkg <- InitPackage()

    points <- as.matrix(points)
    num <- nrow(points)
    dim <- ncol(points)
     
    d <- .C("gprmoverage",
            as.single(points),
            as.integer(num),
            as.integer(dim),
	    as.integer(w),
            d = single(num*dim),
            NAOK = TRUE,
            PACKAGE = pkg)$d

    return (matrix(d,num,dim))
}

#####################################################################
# gpRBasavoff
#
gpRBasavoff <- function(points,w1,w2) {

    pkg <- InitPackage()

    points <- as.matrix(points)
    num <- nrow(points)
    dim <- ncol(points)
     
    d <- .C("gprbasavoff",
            as.single(points),
            as.integer(num),
            as.integer(dim),
	    	as.integer(w1),
	    	as.integer(w2),
            d = single(num*dim),
            NAOK = TRUE,
            PACKAGE = pkg)$d

    return (matrix(d,num,dim))
}

#####################################################################
# gpRBasoroff
#
gpRBasoroff <- function(points,w1) {
	
	pkg <- InitPackage()
	
	points <- as.matrix(points)
	num <- nrow(points)
	dim <- ncol(points)
	
	d <- .C("gprbasoroff",
			as.single(points),
			as.integer(num),
			as.integer(dim),
			as.integer(w1),
			d = single(num*dim),
			NAOK = TRUE,
			PACKAGE = pkg)$d
	
	return (matrix(d,num,dim))
}

#####################################################################
# gpRDiff
#
gpRDiff <- function(points,w) {

    pkg <- InitPackage()

    points <- as.matrix(points)
    num <- nrow(points)
    dim <- ncol(points)
     
    d <- .C("gprdiff",
            as.single(points),
            as.integer(num),
            as.integer(dim),
            as.integer(w),
            d = single(num*dim),
            NAOK = TRUE,
            PACKAGE = pkg)$d

    return (matrix(d,num,dim))
}

#####################################################################
# gpRDiffrev
#
gpRDiffrev <- function(points,w) {
	
	pkg <- InitPackage()
	
	points <- as.matrix(points)
	num <- nrow(points)
	dim <- ncol(points)
	
	d <- .C("gprdiffrev",
			as.single(points),
			as.integer(num),
			as.integer(dim),
			as.integer(w),
			d = single(num*dim),
			NAOK = TRUE,
			PACKAGE = pkg)$d
	
	return (matrix(d,num,dim))
}

#####################################################################
# gpRUp
#
gpRUp <- function(points,w) {
	
	pkg <- InitPackage()
	
	points <- as.matrix(points)
	num <- nrow(points)
	dim <- ncol(points)
	
	d <- .C("gprup",
			as.single(points),
			as.integer(num),
			as.integer(dim),
			as.single(w),
			d = single(num*dim),
			NAOK = TRUE,
			PACKAGE = pkg)$d
	
	return (matrix(d,num,dim))
}

#####################################################################
# gpRDown
#
gpRDown <- function(points,w) {
	
	pkg <- InitPackage()
	
	points <- as.matrix(points)
	num <- nrow(points)
	dim <- ncol(points)
	
	d <- .C("gprdown",
			as.single(points),
			as.integer(num),
			as.integer(dim),
			as.single(w),
			d = single(num*dim),
			NAOK = TRUE,
			PACKAGE = pkg)$d
	
	return (matrix(d,num,dim))
}

#####################################################################
# gpRSdall
#
gpRSdall <- function(points) {
	
	pkg <- InitPackage()
	
	points <- as.matrix(points)
	num <- nrow(points)
	dim <- ncol(points)
	
	d <- .C("gprsdall",
			as.single(points),
			as.integer(num),
			as.integer(dim),
			d = single(1),
			NAOK = TRUE,
			PACKAGE = pkg)$d
	
	return (d)
}


#####################################################################
# gpRAvgall
#
gpRAvgall <- function(points) {
	
	pkg <- InitPackage()
	
	points <- as.matrix(points)
	num <- nrow(points)
	dim <- ncol(points)
	
	d <- .C("gpravgall",
			as.single(points),
			as.integer(num),
			as.integer(dim),
			d = single(1),
			NAOK = TRUE,
			PACKAGE = pkg)$d
	
	return (d)
}

#####################################################################
# gpRMovemax
#
gpRMovemax <- function(points,w) {
	
	pkg <- InitPackage()
	
	points <- as.matrix(points)
	num <- nrow(points)
	dim <- ncol(points)
	
	d <- .C("gprmovemax",
			as.single(points),
			as.integer(num),
			as.integer(dim),
			as.integer(w),
			d = single(num*dim),
			NAOK = TRUE,
			PACKAGE = pkg)$d
	
	return (matrix(d,num,dim))
}


#####################################################################
# gpRMeanmax
#
gpRMeanmax <- function(points,w) {
	
	pkg <- InitPackage()
	
	points <- as.matrix(points)
	num <- nrow(points)
	dim <- ncol(points)
	
	d <- .C("gprmeanmax",
			as.single(points),
			as.integer(num),
			as.integer(dim),
			as.integer(w),
			d = single(num*dim),
			NAOK = TRUE,
			PACKAGE = pkg)$d
	
	return (matrix(d,num,dim))
}


peaks2one<-function(wavg,wmask,wd=100){
	nmask=c()
	ppeaks=which(wmask==1)
	if(length(ppeaks) >= 2){
		pdiff<-ppeaks[2:length(ppeaks)]-ppeaks[1:length(ppeaks)-1]
		if(length(which(pdiff<wd))>0){
			pgood<-ppeaks[-c(which(pdiff<wd)+1)]
			paval<-wavg[ppeaks]
			if(pgood[length(pgood)]!=ppeaks[length(ppeaks)])
				pbord=c(pgood[1:(length(pgood)-1)],ppeaks[length(ppeaks)])
			else pbord=pgood
			for(j in 2:length(pbord)){
				ivec=c(pbord[j-1]:pbord[j])
				if(j>2)	ivec=ivec[2:length(ivec)]
				jvec=ivec[which(wmask[ivec]==1)]
				wvec<-wavg[jvec]
				nmask<-c(nmask,jvec[which(wvec==max(wvec))])
			}
			wmask[-nmask]=0
			ppeaks=nmask
			pdiff<-ppeaks[2:length(ppeaks)]-ppeaks[1:length(ppeaks)-1]
			pgood<-ppeaks[-c(which(pdiff<wd)+1)]
			wmask[-pgood]=0
		}
	}
	wmask
}


#####################################################################
# gpRPeakmask
#
gpRPeakmask <- function(points,w1=80,w2=200,up=1,wgap=100) {
	
    pkg <- InitPackage()
	
	matoutnum<-3
	points <- as.matrix(points)
	num <- nrow(points)
	dim <- ncol(points)
	
	d <- .C("gprpeakmask",
			as.single(points),
			as.integer(num),
			as.integer(dim),
			as.single(w1),
			as.single(w2),
			as.single(up),
			d = single(matoutnum*num*dim),
			NAOK = TRUE,
			PACKAGE = pkg)$d
	
	llmat=matrix(d,num,matoutnum*dim)
	pmask   = llmat[,1:dim]
	pavg = llmat[,(dim+1):(2*dim)]
	for(i in 1:dim)	pmask[,i]<-peaks2one(pavg[,i],pmask[,i],wgap)
	orgoff = llmat[,(2*dim+1):(3*dim)]
	return (list(pmask=pmask, pavg=pavg, orgoff=orgoff))
}


#####################################################################
# gpRPeak2mask
#
gpRPeak2mask <- function(points,w1=45,w2=250,w3=15,wgap=100) {
	
	pkg <- InitPackage()
	
	matoutnum<-4
	
	points <- as.matrix(points)
	num <- nrow(points)
	dim <- ncol(points)
	
	d <- .C("gprpeak2mask",
			as.single(points),
			as.integer(num),
			as.integer(dim),
			as.single(w1),
			as.single(w2),
			as.single(w3),
			d = single(matoutnum*num*dim),
			NAOK = TRUE,
			PACKAGE = pkg)$d
	
	llmat=matrix(d,num,matoutnum*dim)
	pmask   = llmat[,1:dim]
	pavg = llmat[,(dim+1):(2*dim)]
	dmask = llmat[,(2*dim+1):(3*dim)]
	davg = llmat[,(3*dim+1):(4*dim)]
	for(i in 1:dim)	pmask[,i]<-peaks2one(pavg[,i],pmask[,i],wgap)
	for(i in 1:dim)	dmask[,i]<-peaks2one(davg[,i],dmask[,i],wgap)
	return (list(pmask=pmask, pavg=pavg, dmask=dmask, davg=davg))
}

