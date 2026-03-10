library(qtl2)

ncores <- floor(parallel::detectCores()*0.8)

iron <- read_cross2( system.file("extdata", "iron.zip", package="qtl2") )

map <- insert_pseudomarkers(iron$gmap, step=0)

pr <- calc_genoprob(iron, map, cores=ncores)

apr <- genoprob_to_alleleprob(pr)

kinship <- calc_kinship(pr, "loco", cores=ncores)

out <- scan1(pr, iron$pheno, kinship, cores=ncores)


par(mar=c(5.1, 4.1, 1.1, 1.1))
ymx <- maxlod(out) # overall maximum LOD score
plot(out, map, lodcolumn=1, col="slateblue", ylim=c(0, ymx*1.02))
plot(out, map, lodcolumn=2, col="violetred", add=TRUE)
legend("topleft", lwd=2, col=c("slateblue", "violetred"), colnames(out), bg="gray90")

find_peaks(out, map, threshold=4, drop=1.5)

bayes_int(out, map, lodcolumn=2, chr=9, prob=0.95)








