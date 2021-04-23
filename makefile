create_python_env:
	@echo "Create python3.6 env..."
	@sleep 1
	virtualenv --python=/usr/bin/python3.6 ~/fgc
	@echo "Env created at : ~/fgc"

install_python_dep:
	@echo "Install python dep..."
	@sleep 1
	~/fgc/bin/pip3 install install -r requirements.txt

install_ipfs:
	@echo "Install ipfs..."
	@sleep 1
	wget https://github.com/ipfs/go-ipfs/releases/download/v0.7.0/go-ipfs-source.tar.gz
	mkdir -p ${GOPATH}/src/github.com/ipfs/go-ipfs
	tar zxvf go-ipfs-source.tar.gz -C ${GOPATH}/src/github.com/ipfs/go-ipfs
	rm go-ipfs-source.tar.gz
	cd ${GOPATH}/src/github.com/ipfs/go-ipfs && make install

install_go:
	@echo "Install go-1.15.2 ..."
	@sudo echo ""
	@sleep 1
	wget https://dl.google.com/go/go1.15.2.linux-amd64.tar.gz
	tar -xvf go1.15.2.linux-amd64.tar.gz -C /usr/lib/
	rm go1.15.2.linux-amd64.tar.gz
	@echo 'export GOROOT=/usr/lib/go' >> ~/.bashrc
	@echo export GOPATH=${HOME}/go >> ~/.bashrc
	@echo 'export PATH=$GOPATH/bin:$GOROOT/bin:$PATH' >> ~/.bashrc


