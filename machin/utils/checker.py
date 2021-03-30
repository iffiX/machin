from typing import List
import inspect
import torch as t
import torch.nn as nn

from .helper_classes import Counter
from .tensor_board import SummaryWriter


class CheckError(Exception):
    pass


def check_shape(tensor: t.Tensor, required_shape: List[int], name=""):
    """
    Check whether tensor has the specified shape.

    Args:
        tensor: Tensor to check.
        required_shape: A list of ``int`` specifying shape of each dimension.
        name: Name of tensor, will be printed in the error message.

    Raises:
        ``RuntimeError`` if shape of the tensor doesn't match.
    """
    shape = list(tensor.shape)
    if shape != required_shape:
        raise CheckError(
            f"Tensor {name} has invalid shape, "
            f"required shape {required_shape}, actual is {shape}"
        )


def check_nan(tensor: t.Tensor, name=""):
    """
    Check whether tensor has ``nan`` element.

    Args:
        tensor: Tensor to check
        name: Name of tensor, will be printed in the error message.

    Raises:
        ``RuntimeError`` if tensor has any ``nan`` element.
    """
    if t.any(t.isnan(tensor)):
        raise CheckError(f"Tensor {name} contains nan!")


def _add_input_check_hook(
    sub_module, counter, interval, writer, hooks, model, module_name
):
    # Generate a input check hook which calls all sub hooks
    # when invoked by pytorch.
    def check_hook(module, input_):
        with t.no_grad():
            if counter.get() % interval == 0:
                # Get forward function signature.

                # Pytorch will not give us keyword arguments of modules,
                # and users also should not use keywork arguments in forward().
                # So we only need to get the 'args' part.
                input_names = inspect.getfullargspec(module.forward).args
                for input_name, input_value in zip(input_names, input_):
                    for hook in hooks:
                        hook(
                            counter,
                            writer,
                            model,
                            module,
                            module_name + ".input." + input_name,
                            input_value,
                        )

    return sub_module.register_forward_pre_hook(check_hook)


def _add_output_check_hook(
    sub_module, counter, interval, writer, hooks, model, module_name
):
    # Generate a output check hook which calls all sub hooks
    # when invoked by pytorch.
    def check_hook(module, _input, output):
        with t.no_grad():
            if counter.get() % interval == 0:
                # Try to resolve output name, if failed, use
                # index as a substitute output name.
                # Currently, we can only judge output number if output is
                # a tuple.
                if isinstance(output, tuple):
                    default_names = [str(i) for i in range(len(output))]
                else:
                    default_names = ["0"]
                output_names = getattr(module, "_chk_output_names", default_names)
                for output_name, output_value in zip(output_names, output):
                    for hook in hooks:
                        hook(
                            counter,
                            writer,
                            model,
                            module,
                            module_name + ".output." + output_name,
                            output_value,
                        )

    return sub_module.register_forward_hook(check_hook)


def _add_param_check_hook(
    sub_module, counter, interval, writer, hooks, model, module_name
):
    # Generate a param check hook which calls all sub hooks
    # when invoked by pytorch.
    handles = []
    for param_name, param_value in sub_module.named_parameters():

        def check_hook(module, _input, _output):  # pragma: no cover
            with t.no_grad():
                if counter.get() % interval == 0:
                    for hook in hooks:
                        hook(
                            counter,
                            writer,
                            model,
                            module,
                            module_name + ".param." + param_name,
                            param_value,
                        )

        handles.append(sub_module.register_forward_hook(check_hook))
    return handles


def i_chk_nan(_counter, _writer, _model, _module, input_name, input_val):
    """
    Check whether there is any nan element in the input, if input is a tensor.
    """
    if t.is_tensor(input_val):
        check_nan(input_val, input_name)


def i_chk_range(counter, writer, _model, _module, input_name, input_val):
    """
    Compute min, max and mean value of the input, if input is a tensor.
    """
    if t.is_tensor(input_val):
        writer.add_scalars(
            input_name,
            {
                "min": t.min(input_val),
                "max": t.max(input_val),
                "mean": t.mean(input_val),
            },
            counter.get(),
        )
        writer.flush()


def o_chk_nan(_counter, _writer, _model, _module, output_name, output_val):
    """
    Check whether there is any nan element in the output, if input is a tensor.
    """
    if t.is_tensor(output_val):
        check_nan(output_val, output_name)


def o_chk_range(counter, writer, _model, _module, output_name, output_val):
    """
    Compute min, max and mean value of the output, if output is a tensor.
    """
    if t.is_tensor(output_val):
        writer.add_scalars(
            output_name,
            {
                "min": t.min(output_val),
                "max": t.max(output_val),
                "mean": t.mean(output_val),
            },
            counter.get(),
        )
        writer.flush()


def p_chk_nan(
    counter, _writer, _model, _module, param_name, param_val
):  # pragma: no cover
    """
    Check whether there is any nan element in the parameter.
    """
    check_nan(param_val, param_name + f"(backward_count={counter.get()})")


def p_chk_range(
    counter, writer, _model, _module, param_name, param_val
):  # pragma: no cover
    """
    Compute min, max and mean value of the parameter.
    """
    writer.add_scalars(
        param_name,
        {"min": t.min(param_val), "max": t.max(param_val), "mean": t.mean(param_val)},
        counter.get(),
    )
    writer.add_histogram(param_name, param_val, counter.get())
    writer.flush()


def mark_as_atom_module(module):
    """
    Mark module as a atom leaf module, so it can be checked.
    """
    setattr(module, "_chk_is_atom", True)


def mark_module_output(module, output_names: List[str]):
    """
    Mark names of the module output. It will also tell checker
    about the number of outputs.

    Args:
        module: Module to be marked.
        output_names: Name of each output value.
    """
    setattr(module, "_chk_output_names", output_names)


def check_model(
    writer: SummaryWriter,
    model: nn.Module,
    input_check_hooks=(i_chk_nan, i_chk_range),
    output_check_hooks=(o_chk_nan, o_chk_range),
    param_check_hooks=(p_chk_nan, p_chk_range),
    input_check_interval=1,
    output_check_interval=1,
    param_check_interval=100,
    name="",
):
    """
    Check model input, output and parameters using hooks. All hooks (Input,
    output and parameter) check hooks are executed in the forward pass.

    An example::

        model = nn.Linear([100, 100])
        check_model(model)

        # Continue to do whatever you like.
        model(t.zeros([100]))

    Note:
        Only leaf modules will be checked (such as ``nn.Linear`` and not some
        complex neural network modules made of several sub-modules). But you
        can manually control granularity.

    Warning:
        Do not output ``tuple`` in your ``forward()`` function if you have
        output check hooks, otherwise you must specify names for each output.

    Hint:
        You may manually control the check granularity by using
        :func:`.mark_as_atom_module`.

        You may specify a list of names for your module outputs so
        names given to your output check hooks will not be numbers,
        by using :func:`.mark_module_output`

    Hint:
        For all three kinds of hooks, your hook need to have the following
        signature:

        ``hook(counter, writer, model, module, name, value)``

        where:

        - ``counter`` is the :class:`.Counter`, you can use
          :meth:`.Counter.get` to get the current pass number.
        - ``writer`` is :class:`.SummaryWriter` from ``tensorboardx``.
        - ``model`` is your model.
        - ``module`` is the module currently being checked.
        - ``name`` is input/output/parameter name string. For input, their
          detail names will be extracted from module ``forward`` signature.
          Output detail names will be numbers or names you have specified.
        - ``value`` is input/output/parameter value.

    Args:
        writer: Tensorboard ``SummaryWriter`` used to log.
        model: Model to be checked.
        input_check_hooks: A series of input check hooks.
        output_check_hooks: A series of output check hooks.
        param_check_hooks: A series of parameter check hooks.
        input_check_interval: Interval (number of forward passes)
            of input checking.
        output_check_interval: Interval (number of forward passes)
            of output checking.
        param_check_interval: Interval (number of backward passes)
            of parameter checking.
        name: Your model name.

    Returns:
        A function ``f()``, calling ``f()`` will deregister all check hooks.
    """
    handles = []
    forward_counter = Counter()

    def _forward_count(_, __):
        forward_counter.count()

    handles.append(model.register_forward_pre_hook(_forward_count))

    # Register forward & backward checker hooks for all submodules.
    # Input checking are done in forward pre hooks.
    # Param checking are done in backward hooks.

    checked_names = []
    for sub_name, sub_module in model.named_modules(prefix=name):
        sub_module = sub_module  # type: nn.Module

        if len(list(sub_module.modules())) != 1 and not getattr(
            sub_module, "_chk_is_atom", False
        ):
            # Current module has children, not a leaf module, so skip.
            continue
        if any(sub_name.startswith(chk_nm) for chk_nm in checked_names):
            # prevent sub-modules of modules marked as "atom" being checked
            continue
        checked_names.append(sub_name)

        handles.append(
            _add_input_check_hook(
                sub_module,
                forward_counter,
                input_check_interval,
                writer,
                input_check_hooks,
                model,
                sub_name,
            )
        )

        handles.append(
            _add_output_check_hook(
                sub_module,
                forward_counter,
                output_check_interval,
                writer,
                output_check_hooks,
                model,
                sub_name,
            )
        )
        handles += _add_param_check_hook(
            sub_module,
            forward_counter,
            param_check_interval,
            writer,
            param_check_hooks,
            model,
            sub_name,
        )

    def cancel():
        for handle in handles:
            handle.remove()

    return cancel
