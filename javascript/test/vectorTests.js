/* global describe, it */
'use strict';

const expect = require('chai').expect;
const assert = require('chai').assert;
const Vector = require('../lib/Vector');

const v1 = new Vector([1, 0, 0]);
const v2 = new Vector([5, 25, 625]);
const v3 = new Vector([1, 2, 3, 4]);

describe('Vector', () => {
  
  // it('should throw on column number inconsistency', () => {
  //   let error;
  //   try {
  //     new Matrix([[1], [1, 2]]);
  //   }
  //   catch(err) {
  //     error = err;
  //   }
  //   expect(error).to.be.a('error');
  // });
  
});
