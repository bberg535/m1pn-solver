include $(MFEM_CONFIG)
.PHONY: flags compiler mpi
flags:
	@echo '$(MFEM_CPPFLAGS) $(MFEM_INCFLAGS)'
	@echo '$(MFEM_LIBS)'
	@echo '$(MFEM_USE_MPI)'
compiler:
	@echo '$(MFEM_CXX)'
mpi:
	@echo '$(MFEM_USE_MPI)'
