"""Compile Qiskit circuits with the MQT compiler (mqt.core.mlir).

The compiler is imported into this process. mqt.core.mlir binds Qiskit's C
API per minor release, which is why setup/requirements.txt pins the Qiskit
series it was built for; with any other Qiskit the import below fails and
the error says which versions the installed mqt.core accepts.

The compiler target is QFwBackend.target, which is built from the QPM's
telemetry and shared with Qiskit's transpiler. MQT snapshots it with
CompilerTarget.from_qiskit.
"""

from qiskit import QuantumCircuit
from mqt.core import mlir


class MQTCompileError(RuntimeError):
	pass


def _compile_one(circuit, target, device_name):
	# An explicit CompilerTarget only yields executable outputs (OPENQASM3 or
	# QIR) and has no Qiskit output, so read the OpenQASM 3 back as a circuit.
	qasm3 = mlir.compile_program(
		circuit, target=target, output=mlir.OutputFormat.OPENQASM3)
	out = mlir.QCProgram.from_openqasm_str(qasm3.source).to_qiskit()
	# The compiler initialises the output register ("meas = 0") with a
	# classical store. It carries nothing for execution and the OpenQASM 2
	# exporter a QPM may need cannot express it, so drop it.
	out.data = [inst for inst in out.data if inst.operation.name != "store"]
	metadata = dict(circuit.metadata or {})
	metadata.update(out.metadata or {})
	metadata["qfw_compiler"] = "mqt"
	metadata["qfw_compile_device"] = device_name
	out.metadata = metadata
	if not out.name or out.name.startswith("circuit-"):
		out.name = circuit.name
	return out


def mqt_compile(circuits, target):
	"""Compile circuits for the device described by ``target``.

	``target`` is the Qiskit Target provided by QFwBackend.target, whose
	description names the device.
	"""

	single = isinstance(circuits, QuantumCircuit)
	circuit_list = [circuits] if single else list(circuits)
	device_name = target.description
	compiler_target = mlir.CompilerTarget.from_qiskit(target, name=device_name)

	try:
		compiled = [
			_compile_one(circuit, compiler_target, device_name)
			for circuit in circuit_list]
	except Exception as exc:
		raise MQTCompileError(
			f"MQT compile for {device_name} failed: {exc}") from exc
	return compiled[0] if single else compiled
