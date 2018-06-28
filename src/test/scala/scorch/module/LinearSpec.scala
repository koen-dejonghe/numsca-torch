package scorch.module

import ns.Tensor
import org.scalatest.{FlatSpec, Matchers}
import scorch.{Module, Variable}
import scorch.Function._
import scorch.optimizer.SGD

class LinearSpec extends FlatSpec with Matchers {

  "Linear" should "forward/backward" in {

    val x = ns.arange(max = 12).reshape(4, 3)
    val w = ns.arange(max = 15).reshape(5, 3)
    val b = ns.arange(max = 5)

    val weights = Variable(w, name = Some("weights"))
    val bias = Variable(b, name = Some("bias"))
    val input = Variable(x)

    val l = Linear(weights, bias)

    val output: Variable = l.forward(input)

    println(output)

    val dy = Variable(ns.arange(max = 20).reshape(4, 5))

    output.backward(dy)

    println(weights.grad)
    println(bias.grad)

    weights.grad.data isSameAs Tensor(210, 240, 270, 228, 262, 296, 246, 284,
      322, 264, 306, 348, 282, 328, 374).reshape(5, 3) shouldBe true

    bias.grad.data isSameAs Tensor(30, 34, 38, 42, 46) shouldBe true
  }

  it should "handle 2 layers" in {

    val x = Variable(ns.arange(max = 12).reshape(4, 3), name = Some("x"))

    val w1 = Variable(ns.arange(max = 15).reshape(5, 3), name = Some("w1"))
    val b1 = Variable(ns.arange(max = 5), name = Some("b1"))
    val l1 = Linear(w1, b1)

    val w2 = Variable(ns.arange(max = 30).reshape(6, 5), name = Some("w2"))
    val b2 = Variable(ns.arange(max = 6), name = Some("b2"))
    val l2 = Linear(w2, b2)

    val out = x ~> l1 ~> l2
    val dOut = Variable(ns.arange(max = 24).reshape(4, 6))
    out.backward(dOut)

    println(out)
    println(w2.grad)
    println(b2.grad)
    println(w1.grad)
    println(b1.grad)

    assert(
      ns.equal(
        ns.array(350, 976, 1602, 2228, 2854, 3480, 1250, 3451, 5652, 7853,
            10054, 12255, 2150, 5926, 9702, 13478, 17254, 21030, 3050, 8401,
            13752, 19103, 24454, 29805)
          .reshape(4, 6),
        out.data
      ))

    assert(
      ns.equal(
        ns.array(936, 3564, 6192, 8820, 11448, 1010, 3840, 6670, 9500, 12330,
            1084, 4116, 7148, 10180, 13212, 1158, 4392, 7626, 10860, 14094,
            1232, 4668, 8104, 11540, 14976, 1306, 4944, 8582, 12220,
            15858)
          .reshape(6, 5),
        w2.grad.data
      ))

    assert(ns.equal(ns.array(36, 40, 44, 48, 52, 56), b2.grad.data))

    assert(
      ns.equal(ns.array(23850, 27650, 31450, 25632, 29708, 33784, 27414, 31766,
                   36118, 29196, 33824, 38452, 30978, 35882, 40786)
                 .reshape(5, 3),
               w1.grad.data))

    assert(ns.equal(ns.array(3800, 4076, 4352, 4628, 4904), b1.grad.data))

  }

  it should "1 pass of a simple nn" in {

    ns.setSeed(231)
    case class Net() extends Module {
      val fc1 = Linear(25, 100)
      val fc2 = Linear(100, 10)
      override def forward(x: Variable): Variable =
        x ~> fc1 ~> relu ~> fc2
    }

    val net = Net()

    val input = Variable(ns.randn(16, 25))
    val target = Variable(ns.randint(0, 10, List(16)))

    val output = net(input)
    val loss = crossEntropy(output, target)
    loss.backward()
    // println(net.fc1.weights.grad)
  }

  it should "multiple passes of nn" in {

    val numSamples = 160
    val numFeatures = 250
    val numClasses = 10

    ns.setSeed(231)
    case class Net() extends Module {
      val fc1 = Linear(numFeatures, 100)
      val fc2 = Linear(100, numClasses)
      val f = Linear(numFeatures, numClasses)
      override def forward(x: Variable): Variable =
        x ~> f
//         x ~> fc1 ~> relu ~> fc2
//        x ~> f ~> relu
        // Variable(ns.zeros(numSamples, numClasses))
    }

//    val net = Net()

//    val input = Variable(ns.randn(numSamples, numFeatures))
//    val target = Variable(ns.randint(0, numClasses, List(numSamples)))
//
//    val optimizer = SGD(net.parameters, 0.08)

    for (i <- 1 to 10000000) {
      ns.zeros(numSamples, numClasses)

      Thread.sleep(1)
//      net.zeroGrad()
//      val output = net(input)

//      val guessed = ns.argmax(output.data, axis = 1)
//      val accuracy = ns.sum(target.data == guessed) / numSamples

//      val loss = crossEntropy(output, target)
//      loss.backward()
      // println(s"$i: loss: ${loss.value(0)} accuracy: $accuracy")
//      optimizer.step()
    }

  }

}
