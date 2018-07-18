package scorch.module

import org.scalatest.{FlatSpec, Matchers}
import scorch.Variable

class Conv2dSpec extends FlatSpec with Matchers {

  "Conv2d" should "do" in {
    val xShape = List(2, 3, 4, 4)
    val wShape = List(3, 3, 4, 4)
    val x =
      Variable(ns.linspace(-0.1, 0.5, steps = xShape.product).reshape(xShape))
    val w =
      Variable(ns.linspace(-0.2, 0.3, steps = wShape.product).reshape(wShape))
    val b = Variable(ns.linspace(-0.1, 0.2, steps = 3))
    val pad = 1
    val stride = 2

    val fInput = Variable(w.data.copy())

    val out = Conv2d(w, b, fInput, 3, 3, stride, stride, pad, pad, 1.0).forward(x)

    val correctOut = ns
      .array(-0.08759809, -0.10987781, -0.18387192, -0.2109216, 0.21027089,
        0.21661097, 0.22847626, 0.23004637, 0.50813986, 0.54309974, 0.64082444,
        0.67101435, -0.98053589, -1.03143541, -1.19128892, -1.24695841,
        0.69108355, 0.66880383, 0.59480972, 0.56776003, 2.36270298, 2.36904306,
        2.38090835, 2.38247847)
      .reshape(2, 3, 2, 2)

    println(out)

  }

  it should "convolve" in {
    val m = Conv2d(16, 33, 3, stride=2, pad=0)

    println(m.weight.shape)
    println(m.bias.shape)
    val input = Variable(ns.randn(20, 16, 50, 100))

    val output = m.forward(input)

    println(output.shape)

    output.backward()
  }

  it should "convolve 2" in {
    val m = Conv2d(3, 6, 5, 2, 1)

    println(m.weight.shape)
    println(m.bias.shape)
    val input = Variable(ns.randn(2, 3, 4, 4))

    val output = m.forward(input)

    println(output.shape)

    output.backward()
  }

}
