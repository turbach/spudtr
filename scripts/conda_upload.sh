# spudtr conda uploader. Runs but doesn't attempt the upload unless on
# master branch with a major.minor.patch version and a set $ANACONDA_TOKEN

# some guarding ...
if [[ -z ${CONDA_DEFAULT_ENV} ]]; then
    echo "activate a conda env before running conda_upload.sh"
    exit -1
fi

# intended for TravisCI deploy but can be tricked into running locally
if [[ "$TRAVIS" != "true" || -z "$TRAVIS_BRANCH" || -z "${PACKAGE_NAME}" ]]; then
    echo "conda_upload.sh is meant to run on TravisCI, if you know what you are doing, fake it locally like so:"
    echo 'export PACKAGE_NAME="spudtr"; export TRAVIS="true"; export TRAVIS_BRANCH="a_branch_name"' 
    exit -2
fi

# set parent of conda-bld, the else isn't needed for travis, simplifies local testing
if [ $USER = "travis" ]; then
    bld_prefix="/home/travis/miniconda"  # from the .travis.yml
else
    bld_prefix=${CONDA_PREFIX}
fi

# on travis there should be a single linux-64 package tarball. insist
tarball=`/bin/ls -1 ${bld_prefix}/conda-bld/linux-64/${PACKAGE_NAME}-*-*.tar.bz2`
n_tarballs=`echo "${tarball}" | wc -w`
if (( $n_tarballs != 1 )); then
    echo "found $n_tarballs package tarballs there must be exactly 1"
    echo "$tarball"
    exit -3
fi

# version string from spudtr/__init__.py and the conda meta.yaml {% version = any_stringr %}
version=`echo $tarball | sed -n "s/.*${PACKAGE_NAME}-[v]\{0,1\}\(.\+\)-.*/\1/p"`

# extract the major.minor.patch of version
mmp=`echo $version | sed -n "s/\(\([0-9]\+\.\)\{1,2\}[0-9]\+\).*/\1/p"`

# toggle whether this is a release version
if [[ "${version}" = "$mmp" ]]; then
    is_release_ver="true"
else
    is_release_ver="false"
fi

# thus far ...
echo "travis branch: $TRAVIS_BRANCH"
echo "package name: $PACKAGE_NAME"
echo "conda-bld: ${bld_prefix}/conda-bld/linux-64"
echo "tarball: $tarball"
echo "conda meta.yaml version: $version"
echo "is_release_ver: $is_release_ver"
echo "Anaconda.org upload command ..."

conda_cmd="anaconda --token $ANACONDA_TOKEN upload ${tarball}"
echo "conda upload command: ${conda_cmd}"

# POSIX trick sets an unset or empty string $ANACONDA_TOKEN to a default string "[not_set]"
ANACONDA_TOKEN=${ANACONDA_TOKEN:-[not_set]}

# attempt the upload if there is token and branch is master with version string major.minor.patch
# else, status report and exit happily
if [[ $ANACONDA_TOKEN != "[not_set]" && $TRAVIS_BRANCH = "master" ]]; then

    # require major.minor.patch version strings for conda upload
    if [[ $is_release_ver = "false" ]]; then
	echo "Version string error $PACKAGE_NAME ${version}: on the master branch, the version string must be major.minor.patch"
	exit -4
    fi

    echo "Attempting upload to Anconda Cloud $PACKAGE_NAME$ $version"
    if ${conda_cmd}; then
	echo "OK"
    else
	echo "Failed"
	exit -5
    fi
else
    echo "$PACKAGE_NAME $TRAVIS_BRANCH $version conda_upload.sh dry run OK"
fi
exit 0
