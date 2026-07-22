# Neural Network for XOR problem using pure Python
import math
import random

from propagator import (
    Cell,
    adder,
    constant,
    divider,
    bring_in,
    function_to_propagator_constructor,
    initialize_scheduler,
    make_tms,
    multiplier,
    nothing_p,
    run,
    subtractor,
    kick_out,
    tms_query,
)
from propagator.merge import make_generic_operator
from propagator.supported_values import Supported, supported, supported_p, supported_unpacking
from propagator.tms import Tms, full_tms_unpacking, tms_p

# =============================================================================
# PURE PYTHON VERSION
# =============================================================================


def sigmoid(x: float) -> float:
    """Sigmoid activation function using pure Python."""
    # Clamp to avoid overflow
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x: float) -> float:
    """Derivative of sigmoid (assumes x is already sigmoid output)."""
    return x * (1 - x)


def dot_product(vec1: list[float], vec2: list[float]) -> float:
    """Compute dot product of two vectors."""
    return sum(a * b for a, b in zip(vec1, vec2))


def matrix_vector_mult(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Multiply a matrix by a vector. Matrix is list of rows."""
    return [dot_product(row, vector) for row in matrix]


def outer_product(vec1: list[float], vec2: list[float]) -> list[list[float]]:
    """Compute outer product of two vectors, returning a matrix."""
    return [[a * b for b in vec2] for a in vec1]


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    """Transpose a matrix (list of lists)."""
    if not matrix:
        return []
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]


def matrix_add(m1: list[list[float]], m2: list[list[float]]) -> list[list[float]]:
    """Add two matrices element-wise."""
    return [[m1[i][j] + m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))]


def scalar_mult_matrix(scalar: float, matrix: list[list[float]]) -> list[list[float]]:
    """Multiply a matrix by a scalar."""
    return [[scalar * val for val in row] for row in matrix]


def _sigmoid_value(x: float | None) -> float | None:
    if x is None:
        return None
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))


generic_sigmoid = make_generic_operator(1, "sigmoid", _sigmoid_value)
generic_sigmoid.assign_operation(supported_unpacking(generic_sigmoid), supported_p)
generic_sigmoid.assign_operation(full_tms_unpacking(generic_sigmoid), tms_p)
sigmoid_propagator = function_to_propagator_constructor(generic_sigmoid)


def _cell_value(content: object) -> object:
    if isinstance(content, Tms):
        result = tms_query(content)
        if isinstance(result, Supported):
            return result.value
        return result
    if isinstance(content, Supported):
        return content.value
    return content


def _cell_support_labels(content: object) -> list[object]:
    if isinstance(content, Tms):
        result = tms_query(content)
        if isinstance(result, Supported):
            labels = [wrapper.obj for wrapper in result.support]
            return _unique_preserving_order(labels)
        return []
    if isinstance(content, Supported):
        labels = [wrapper.obj for wrapper in content.support]
        return _unique_preserving_order(labels)
    return []


def _unique_preserving_order(values: list[object]) -> list[object]:
    seen = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _make_tagged_cell(name: str, value: float, premise: str) -> Cell:
    cell = Cell(name=name)
    cell.add_content(make_tms(supported(value, [premise])))
    return cell


class NeuralNetworkPython:
    """
    A simple neural network with one hidden layer for solving the XOR problem.
    Uses only pure Python data structures (lists) - no NumPy required.

    Architecture:
    - Input layer: 2 neurons
    - Hidden layer: configurable size (default 2)
    - Output layer: 1 neuron
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        # Initialize weights randomly
        # weights_input_hidden: input_size x hidden_size matrix
        self.weights_input_hidden = [
            [random.random() for _ in range(hidden_size)] for _ in range(input_size)
        ]
        # weights_hidden_output: hidden_size x output_size matrix
        self.weights_hidden_output = [
            [random.random() for _ in range(output_size)] for _ in range(hidden_size)
        ]

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, inputs: list[float]) -> list[float]:
        """
        Forward pass: compute output from inputs.

        Args:
            inputs: list of input values (length = input_size)

        Returns:
            list of output values (length = output_size)
        """
        # Input to hidden layer: multiply inputs by weights, apply activation
        hidden_input = matrix_vector_mult(transpose(self.weights_input_hidden), inputs)
        hidden_output = [sigmoid(x) for x in hidden_input]

        # Hidden to output layer
        output_input = matrix_vector_mult(
            transpose(self.weights_hidden_output), hidden_output
        )
        output = [sigmoid(x) for x in output_input]

        return output

    def train(
        self,
        inputs_batch: list[list[float]],
        targets_batch: list[list[float]],
        epochs: int,
        learning_rate: float = 1.0,
    ) -> list[float]:
        """
        Train the network using backpropagation.

        Args:
            inputs_batch: list of input samples, each sample is a list
            targets_batch: list of target outputs, each target is a list
            epochs: number of training iterations
            learning_rate: step size for weight updates

        Returns:
            loss_history: list of MSE loss values for each epoch
        """
        loss_history = []

        for _ in range(epochs):
            total_error_squared = 0

            # Accumulate gradients for batch
            grad_hidden_output = [
                [0.0] * self.output_size for _ in range(self.hidden_size)
            ]
            grad_input_hidden = [
                [0.0] * self.hidden_size for _ in range(self.input_size)
            ]

            for inputs, targets in zip(inputs_batch, targets_batch):
                # Forward pass
                hidden_input = matrix_vector_mult(
                    transpose(self.weights_input_hidden), inputs
                )
                hidden_output = [sigmoid(x) for x in hidden_input]

                output_input = matrix_vector_mult(
                    transpose(self.weights_hidden_output), hidden_output
                )
                output = [sigmoid(x) for x in output_input]

                # Calculate error
                error = [t - o for t, o in zip(targets, output)]
                total_error_squared += sum(e * e for e in error)

                # Backpropagation - output layer
                d_output = [e * sigmoid_derivative(o) for e, o in zip(error, output)]

                # Backpropagation - hidden layer
                # error_hidden = d_output * weights_hidden_output^T
                error_hidden = []
                for h in range(self.hidden_size):
                    err = sum(
                        d_output[o] * self.weights_hidden_output[h][o]
                        for o in range(self.output_size)
                    )
                    error_hidden.append(err)

                d_hidden = [
                    eh * sigmoid_derivative(ho)
                    for eh, ho in zip(error_hidden, hidden_output)
                ]

                # Accumulate gradients
                # grad_hidden_output += outer_product(hidden_output, d_output)
                for h in range(self.hidden_size):
                    for o in range(self.output_size):
                        grad_hidden_output[h][o] += hidden_output[h] * d_output[o]

                # grad_input_hidden += outer_product(inputs, d_hidden)
                for i in range(self.input_size):
                    for h in range(self.hidden_size):
                        grad_input_hidden[i][h] += inputs[i] * d_hidden[h]

            # Update weights (using accumulated gradients)
            for h in range(self.hidden_size):
                for o in range(self.output_size):
                    self.weights_hidden_output[h][o] += (
                        learning_rate * grad_hidden_output[h][o]
                    )

            for i in range(self.input_size):
                for h in range(self.hidden_size):
                    self.weights_input_hidden[i][h] += (
                        learning_rate * grad_input_hidden[i][h]
                    )

            # Track loss (Mean Squared Error)
            loss = total_error_squared / len(inputs_batch)
            loss_history.append(loss)

        return loss_history



class NeuralNetworkPropagator:
    """
    A reactive propagator neural network for XOR.

    Everything that carries model state or training data is represented as
    Cells, including weights, inputs, targets, epoch count, learning rate,
    and the training trigger itself. Training starts when content is added to
    the training trigger cell.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        seed: int | None = None,
    ):
        initialize_scheduler()

        if seed is not None:
            random.seed(seed)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.learning_rate_cell = Cell(name="learning_rate")
        self.learning_rate_cell.add_content(make_tms(supported(1.0, ["learning-rate"])))

        self.epoch_count_cell = Cell(name="epoch_count")
        self.epoch_count_cell.add_content(make_tms(supported(1, ["epoch-count"])))

        self.train_trigger_cell = Cell(name="train_trigger")

        self.training_input_cells: list[list[Cell]] = []
        self.training_target_cells: list[list[Cell]] = []

        self.initial_weight_cells_input_hidden = [
            [Cell(name=f"w_ih_init_{i}_{h}") for h in range(hidden_size)]
            for i in range(input_size)
        ]
        self.initial_weight_cells_hidden_output = [
            [Cell(name=f"w_ho_init_{h}_{o}") for o in range(output_size)]
            for h in range(hidden_size)
        ]

        for i in range(input_size):
            for h in range(hidden_size):
                value = random.random()
                premise = f"initial-w-ih-{i}-{h}"
                self.initial_weight_cells_input_hidden[i][h].add_content(
                    make_tms(supported(value, [premise]))
                )

        for h in range(hidden_size):
            for o in range(output_size):
                value = random.random()
                premise = f"initial-w-ho-{h}-{o}"
                self.initial_weight_cells_hidden_output[h][o].add_content(
                    make_tms(supported(value, [premise]))
                )

        self.trained_weight_cells_input_hidden = [
            row[:] for row in self.initial_weight_cells_input_hidden
        ]
        self.trained_weight_cells_hidden_output = [
            row[:] for row in self.initial_weight_cells_hidden_output
        ]

        self.last_loss_cell = Cell(name="last_loss")
        self.last_epoch_loss_cells: list[Cell] = []

        self.last_weight_provenance = {
            "input_hidden": [[[] for _ in range(hidden_size)] for _ in range(input_size)],
            "hidden_output": [[[] for _ in range(output_size)] for _ in range(hidden_size)],
        }

        from propagator.cell import propagator as register_propagator

        register_propagator([self.train_trigger_cell], self._on_train_trigger)

    def _epoch_count(self, epochs: object) -> int:
        def _safe_int(value: object, default: int = 1) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        if isinstance(epochs, int):
            return epochs
        if isinstance(epochs, float) and epochs.is_integer():
            return int(epochs)
        if isinstance(epochs, Cell):
            return _safe_int(_cell_value(epochs.content))
        if isinstance(epochs, Supported):
            return _safe_int(epochs.value)
        if isinstance(epochs, Tms):
            resolved = tms_query(epochs)
            if isinstance(resolved, Supported):
                return _safe_int(resolved.value)
            if resolved is not None:
                return _safe_int(resolved)
        return _safe_int(epochs)

    def set_training_data(
        self,
        inputs_batch: list[list[float]],
        targets_batch: list[list[float]],
    ) -> None:
        self.training_input_cells = []
        self.training_target_cells = []

        for sample_index, inputs in enumerate(inputs_batch):
            sample_row: list[Cell] = []
            for i, value in enumerate(inputs):
                cell = Cell(name=f"training-sample-{sample_index}_input_{i}")
                cell.add_content(
                    make_tms(supported(value, [f"training-sample-{sample_index}"]))
                )
                sample_row.append(cell)
            self.training_input_cells.append(sample_row)

        for sample_index, targets in enumerate(targets_batch):
            sample_row = []
            for o, value in enumerate(targets):
                cell = Cell(name=f"training-sample-{sample_index}_target_{o}")
                cell.add_content(
                    make_tms(supported(value, [f"training-sample-{sample_index}"]))
                )
                sample_row.append(cell)
            self.training_target_cells.append(sample_row)

    def set_learning_rate(self, learning_rate: float, premise: str = "learning-rate") -> None:
        self.learning_rate_cell = Cell(name="learning_rate")
        self.learning_rate_cell.add_content(make_tms(supported(learning_rate, [premise])))

    def set_epoch_count(
        self,
        epochs: int | float | Supported | Tms | Cell,
        premise: str = "epoch-count",
    ) -> None:
        epoch_count = self._epoch_count(epochs)
        self.epoch_count_cell = Cell(name="epoch_count")
        self.epoch_count_cell.add_content(make_tms(supported(epoch_count, [premise])))

    def request_training(self, run_tag: str, premise: str = "train-request") -> None:
        trigger_premise = f"{premise}:{run_tag}"
        self.train_trigger_cell.add_content(make_tms(supported(True, [trigger_premise])))
        run()

    def _training_premises(self, sample_count: int) -> list[str]:
        return [f"training-sample-{sample_index}" for sample_index in range(sample_count)]

    def _active_sample_indices(
        self,
        epoch_index: int,
    ) -> list[int]:
        active_sample_indices = []
        for sample_index, (input_cells, target_cells) in enumerate(
            zip(self.training_input_cells, self.training_target_cells)
        ):
            input_probe = input_cells[0]
            target_probe = target_cells[0]
            if not nothing_p(tms_query(input_probe.content)) and not nothing_p(tms_query(target_probe.content)):
                active_sample_indices.append(sample_index)
        return active_sample_indices

    def _sum_cells(self, name: str, cells: list[Cell]) -> Cell:
        total = Cell(name=f"{name}_init")
        constant(0.0, total)
        current_total = total
        for index, cell in enumerate(cells):
            next_total = Cell(name=f"{name}_{index}")
            adder(current_total, cell, next_total)
            current_total = next_total
        return current_total

    def _build_forward_and_backward_graph(
        self,
        weight_cells_input_hidden: list[list[Cell]],
        weight_cells_hidden_output: list[list[Cell]],
        epoch_index: int,
    ) -> tuple[list[list[Cell]], list[list[Cell]], Cell]:
        learning_rate_cell = self.learning_rate_cell

        sample_losses = []
        input_gradient_terms = [
            [[] for _ in range(self.hidden_size)] for _ in range(self.input_size)
        ]
        hidden_output_gradient_terms = [
            [[] for _ in range(self.output_size)] for _ in range(self.hidden_size)
        ]

        sample_premises = self._training_premises(len(self.training_input_cells))
        active_sample_indices = self._active_sample_indices(epoch_index)

        for sample_index, (input_cells, target_cells) in enumerate(
            zip(self.training_input_cells, self.training_target_cells)
        ):
            if sample_index not in active_sample_indices:
                continue

            sample_tag = sample_premises[sample_index]

            hidden_outputs = []
            for h in range(self.hidden_size):
                current_sum = Cell(name=f"{sample_tag}_hidden_sum_{h}_init")
                constant(0.0, current_sum)

                for i in range(self.input_size):
                    product_cell = Cell(name=f"{sample_tag}_hidden_prod_{i}_{h}")
                    multiplier(input_cells[i], weight_cells_input_hidden[i][h], product_cell)

                    next_sum = Cell(name=f"{sample_tag}_hidden_sum_{h}_{i}")
                    adder(current_sum, product_cell, next_sum)
                    current_sum = next_sum

                hidden_output = Cell(name=f"{sample_tag}_hidden_output_{h}")
                sigmoid_propagator(current_sum, hidden_output)
                hidden_outputs.append(hidden_output)

            output_cells = []
            for o in range(self.output_size):
                current_sum = Cell(name=f"{sample_tag}_output_sum_{o}_init")
                constant(0.0, current_sum)

                for h in range(self.hidden_size):
                    product_cell = Cell(name=f"{sample_tag}_output_prod_{h}_{o}")
                    multiplier(hidden_outputs[h], weight_cells_hidden_output[h][o], product_cell)

                    next_sum = Cell(name=f"{sample_tag}_output_sum_{o}_{h}")
                    adder(current_sum, product_cell, next_sum)
                    current_sum = next_sum

                output_cell = Cell(name=f"{sample_tag}_output_{o}")
                sigmoid_propagator(current_sum, output_cell)
                output_cells.append(output_cell)

            sample_squared_errors = []
            output_deltas = []
            for o in range(self.output_size):
                error_cell = Cell(name=f"{sample_tag}_error_{o}")
                subtractor(target_cells[o], output_cells[o], error_cell)

                squared_error = Cell(name=f"{sample_tag}_squared_error_{o}")
                multiplier(error_cell, error_cell, squared_error)
                sample_squared_errors.append(squared_error)

                one_cell = Cell(name=f"{sample_tag}_one_for_output_{o}")
                constant(1.0, one_cell)

                one_minus_output = Cell(name=f"{sample_tag}_one_minus_output_{o}")
                subtractor(one_cell, output_cells[o], one_minus_output)

                output_derivative = Cell(name=f"{sample_tag}_output_derivative_{o}")
                multiplier(output_cells[o], one_minus_output, output_derivative)

                output_delta = Cell(name=f"{sample_tag}_output_delta_{o}")
                multiplier(error_cell, output_derivative, output_delta)
                output_deltas.append(output_delta)

            sample_loss = Cell(name=f"{sample_tag}_loss_init")
            constant(0.0, sample_loss)
            current_loss = sample_loss
            for o, squared_error in enumerate(sample_squared_errors):
                next_loss = Cell(name=f"{sample_tag}_loss_{o}")
                adder(current_loss, squared_error, next_loss)
                current_loss = next_loss
            sample_losses.append(current_loss)

            hidden_deltas = []
            for h in range(self.hidden_size):
                current_hidden_error = Cell(name=f"{sample_tag}_hidden_error_{h}_init")
                constant(0.0, current_hidden_error)

                for o in range(self.output_size):
                    contribution = Cell(name=f"{sample_tag}_hidden_error_term_{h}_{o}")
                    multiplier(output_deltas[o], weight_cells_hidden_output[h][o], contribution)

                    next_hidden_error = Cell(name=f"{sample_tag}_hidden_error_{h}_{o}")
                    adder(current_hidden_error, contribution, next_hidden_error)
                    current_hidden_error = next_hidden_error

                one_cell = Cell(name=f"{sample_tag}_one_for_hidden_{h}")
                constant(1.0, one_cell)

                one_minus_hidden = Cell(name=f"{sample_tag}_one_minus_hidden_{h}")
                subtractor(one_cell, hidden_outputs[h], one_minus_hidden)

                hidden_derivative = Cell(name=f"{sample_tag}_hidden_derivative_{h}")
                multiplier(hidden_outputs[h], one_minus_hidden, hidden_derivative)

                hidden_delta = Cell(name=f"{sample_tag}_hidden_delta_{h}")
                multiplier(current_hidden_error, hidden_derivative, hidden_delta)
                hidden_deltas.append(hidden_delta)

            for h in range(self.hidden_size):
                for o in range(self.output_size):
                    grad_cell = Cell(name=f"{sample_tag}_grad_ho_{h}_{o}")
                    multiplier(hidden_outputs[h], output_deltas[o], grad_cell)
                    hidden_output_gradient_terms[h][o].append(grad_cell)

            for i in range(self.input_size):
                for h in range(self.hidden_size):
                    grad_cell = Cell(name=f"{sample_tag}_grad_ih_{i}_{h}")
                    multiplier(input_cells[i], hidden_deltas[h], grad_cell)
                    input_gradient_terms[i][h].append(grad_cell)

        updated_weights_input_hidden = []
        for i in range(self.input_size):
            row = []
            for h in range(self.hidden_size):
                gradient_sum = self._sum_cells(f"grad_ih_{i}_{h}", input_gradient_terms[i][h])
                scaled_gradient = Cell(name=f"scaled_grad_ih_{i}_{h}")
                multiplier(learning_rate_cell, gradient_sum, scaled_gradient)

                updated_weight = Cell(name=f"updated_w_ih_{i}_{h}")
                adder(weight_cells_input_hidden[i][h], scaled_gradient, updated_weight)
                row.append(updated_weight)
            updated_weights_input_hidden.append(row)

        updated_weights_hidden_output = []
        for h in range(self.hidden_size):
            row = []
            for o in range(self.output_size):
                gradient_sum = self._sum_cells(f"grad_ho_{h}_{o}", hidden_output_gradient_terms[h][o])
                scaled_gradient = Cell(name=f"scaled_grad_ho_{h}_{o}")
                multiplier(learning_rate_cell, gradient_sum, scaled_gradient)

                updated_weight = Cell(name=f"updated_w_ho_{h}_{o}")
                adder(weight_cells_hidden_output[h][o], scaled_gradient, updated_weight)
                row.append(updated_weight)
            updated_weights_hidden_output.append(row)

        epoch_loss = self._sum_cells("epoch_loss", sample_losses)

        return updated_weights_input_hidden, updated_weights_hidden_output, epoch_loss

    def _on_train_trigger(self) -> None:
        trigger_value = _cell_value(self.train_trigger_cell.content)
        if trigger_value is None:
            return
        if not self.training_input_cells or not self.training_target_cells:
            return

        epoch_count = max(1, self._epoch_count(self.epoch_count_cell))
        weight_cells_input_hidden = [
            row[:] for row in self.initial_weight_cells_input_hidden
        ]
        weight_cells_hidden_output = [
            row[:] for row in self.initial_weight_cells_hidden_output
        ]

        epoch_losses = []
        for epoch_index in range(epoch_count):
            (
                weight_cells_input_hidden,
                weight_cells_hidden_output,
                epoch_loss,
            ) = self._build_forward_and_backward_graph(
                weight_cells_input_hidden,
                weight_cells_hidden_output,
                epoch_index,
            )
            epoch_losses.append(epoch_loss)

        self.trained_weight_cells_input_hidden = weight_cells_input_hidden
        self.trained_weight_cells_hidden_output = weight_cells_hidden_output
        self.last_epoch_loss_cells = epoch_losses
        self.last_loss_cell = epoch_losses[-1]

        self.last_weight_provenance = {
            "input_hidden": [
                [_cell_support_labels(cell.content) for cell in row]
                for row in self.trained_weight_cells_input_hidden
            ],
            "hidden_output": [
                [_cell_support_labels(cell.content) for cell in row]
                for row in self.trained_weight_cells_hidden_output
            ],
        }

    def forward(self, inputs: list[float], premise_tag: str = "forward-input") -> list[float]:

        input_cells = []
        for i, value in enumerate(inputs):
            cell = Cell(name=f"input_{i}")
            cell.add_content(make_tms(supported(value, [premise_tag])))
            input_cells.append(cell)

        weight_cells_input_hidden = self.trained_weight_cells_input_hidden
        weight_cells_hidden_output = self.trained_weight_cells_hidden_output

        hidden_outputs = []
        for h in range(self.hidden_size):
            current_sum = Cell(name=f"forward_hidden_sum_{h}_init")
            constant(0.0, current_sum)

            for i in range(self.input_size):
                product_cell = Cell(name=f"forward_hidden_prod_{i}_{h}")
                multiplier(input_cells[i], weight_cells_input_hidden[i][h], product_cell)

                next_sum = Cell(name=f"forward_hidden_sum_{h}_{i}")
                adder(current_sum, product_cell, next_sum)
                current_sum = next_sum

            hidden_output = Cell(name=f"forward_hidden_{h}")
            sigmoid_propagator(current_sum, hidden_output)
            hidden_outputs.append(hidden_output)

        output_cells = []
        for o in range(self.output_size):
            current_sum = Cell(name=f"forward_output_sum_{o}_init")
            constant(0.0, current_sum)

            for h in range(self.hidden_size):
                product_cell = Cell(name=f"forward_output_prod_{h}_{o}")
                multiplier(hidden_outputs[h], weight_cells_hidden_output[h][o], product_cell)

                next_sum = Cell(name=f"forward_output_sum_{o}_{h}")
                adder(current_sum, product_cell, next_sum)
                current_sum = next_sum

            output_cell = Cell(name=f"forward_output_{o}")
            sigmoid_propagator(current_sum, output_cell)
            output_cells.append(output_cell)

        run()

        return [_cell_value(cell.content) for cell in output_cells]

    def train(
        self,
        inputs_batch: list[list[float]],
        targets_batch: list[list[float]],
        epochs: int | float | Supported | Tms | Cell,
        learning_rate: float = 1.0,
    ) -> list[float]:
        self.set_training_data(inputs_batch, targets_batch)
        self.set_learning_rate(learning_rate)
        self.set_epoch_count(epochs)
        self.request_training(run_tag=f"train-run-{random.random()}")
        return [_cell_value(cell.content) for cell in self.last_epoch_loss_cells]

    def recompute(
        self,
        inputs_batch: list[list[float]],
        targets_batch: list[list[float]],
        epochs: int | float | Supported | Tms | Cell = 1,
        learning_rate: float = 1.0,
    ) -> float:
        loss_history = self.train(inputs_batch, targets_batch, epochs=epochs, learning_rate=learning_rate)
        return loss_history[-1]

    def believed_weights(self) -> tuple[list[list[float]], list[list[float]]]:
        return (
            [
                [_cell_value(cell.content) for cell in row]
                for row in self.trained_weight_cells_input_hidden
            ],
            [
                [_cell_value(cell.content) for cell in row]
                for row in self.trained_weight_cells_hidden_output
            ],
        )

    def weight_provenance(self) -> dict[str, list[list[list[object]]]]:
        return {
            "input_hidden": [
                [
                    _unique_preserving_order(_cell_support_labels(cell.content))
                    for cell in row
                ]
                for row in self.trained_weight_cells_input_hidden
            ],
            "hidden_output": [
                [
                    _unique_preserving_order(_cell_support_labels(cell.content))
                    for cell in row
                ]
                for row in self.trained_weight_cells_hidden_output
            ],
        }


def demo_propagator_training() -> None:
    print("=" * 60)
    print("XOR Neural Network - Propagator Training Demo")
    print("=" * 60)

    random.seed(42)
    nn = NeuralNetworkPropagator(input_size=2, hidden_size=2, output_size=1, seed=42)
    training_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    training_targets = [[0], [1], [1], [0]]

    nn.set_training_data(training_inputs, training_targets)
    nn.set_learning_rate(1.0)
    nn.set_epoch_count(10000)
    nn.request_training(run_tag="demo-train-0")
    loss_history = [_cell_value(cell.content) for cell in nn.last_epoch_loss_cells]

    print(f"Final loss: {loss_history[-1]:.6f}")
    print("Predictions:")
    for inputs, expected in zip(training_inputs, training_targets):
        predicted = nn.forward(inputs)
        print(f"  {inputs} -> {predicted[0]:.6f} (expected {expected[0]})")

    provenance = nn.weight_provenance()
    print("\nWeight provenance (sample tags that influenced each learned weight):")
    print(f"  input->hidden[0][0]: {provenance['input_hidden'][0][0]}")
    print(f"  hidden->output[0][0]: {provenance['hidden_output'][0][0]}")

    print("\nTriggering a worldview change and retraining one epoch:")
    kick_out("training-sample-3")
    nn.set_epoch_count(100)
    nn.request_training(run_tag="demo-train-kick")
    kicked_loss = _cell_value(nn.last_loss_cell.content)
    print(f"  loss after kick_out retrain: {kicked_loss:.6f}")
    print(f"  output after kick_out retrain: {nn.forward([1, 0])[0]:.6f}")

    bring_in("training-sample-3")
    nn.request_training(run_tag="demo-train-restore")
    restored_loss = _cell_value(nn.last_loss_cell.content)
    print(f"  loss after bring_in retrain: {restored_loss:.6f}")
    print(f"  output after bring_in retrain: {nn.forward([1, 0])[0]:.6f}")


if __name__ == "__main__":
    demo_propagator_training()
