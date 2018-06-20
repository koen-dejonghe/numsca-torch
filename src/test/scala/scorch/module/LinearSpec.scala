package scorch.module

import org.scalatest.{FlatSpec, Matchers}
import scorch.Variable

class LinearSpec extends FlatSpec with Matchers {

  "Linear" should "forward" in {

    val x = ns.arange(max = 12).reshape(List(4, 3))
    val w = ns.arange(max = 80).reshape(20, 3)
    val b = ns.arange(max = 20)

    val weights = Variable(w, name = Some("weights"))
    val bias = Variable(b, name = Some("bias"))
    val input = Variable(x)

    val l = Linear(weights, bias)

    val output: Variable = l.forward(input)

    println(output)

    val dy = Variable(ns.randn(4, 20))

    output.backward(dy)

    println(weights.grad)
    println(bias.grad)

  }

}
